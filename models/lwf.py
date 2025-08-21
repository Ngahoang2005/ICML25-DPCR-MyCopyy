import logging
import numpy as np
import torch
import os
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from utils.data_manager import DummyDataset
from utils.inc_net import IncrementalNet, CosineIncrementalNet, Drift_Estimator, ALClassifier
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from torchvision import datasets, transforms
from utils.autoaugment import CIFAR10Policy

init_epoch = 2
init_lr = 0.1 
init_milestones = [60, 120, 160]
init_lr_decay = 0.1
init_weight_decay = 0.0005

# cifar100
epochs = 2 
lrate = 0.05 
milestones = [45, 90]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2
lamda = 10

EPSILON = 1e-8

class LwF(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        if self.args["dataset"] == "imagenet100" or self.args["dataset"] == "imagenet1000":
            epochs = 100
            lrate = 0.05
            milestones = [45, 90]
            lrate_decay = 0.1
            batch_size = 128
            weight_decay = 2e-4
            num_workers = 8
            T = 2
            lamda = 5
            self.num_per_class = 1300
        elif self.args["dataset"] == "tinyimagenet200":
            epochs = 100
            lrate = 0.001
            milestones = [45, 90]
            lrate_decay = 0.1
            batch_size = 128
            weight_decay = 2e-4
            num_workers = 8
            T = 2
            lamda = 10
        print("Number of samples per class:{}".format(self.num_per_class))
        if self.args["dataset"] == "cub200":
            init_lr = 0.1
            lrate = 0.05
            lamda = 20
            self.num_per_class = 30
        if self.args["cosine"]:
            self._network = CosineIncrementalNet(args, False)
        else:
            self._network = IncrementalNet(args, False)

        self._protos = []
        self.al_classifier = None
        if self.args["DPCR"]:
            self._covs = []
            self._projectors = []
        # không in thông tin model ở đây để sạch log
        self.old_task_mem = None    # gradient-memory dùng cho GPM / projection
        self._network_proj = None   # sẽ chứa bản gradient-projected network khi train task>0

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        if not self.args['resume']:
            if not os.path.exists(self.args["model_dir"]):
                os.makedirs(self.args["model_dir"])
            self.save_checkpoint("{}".format(self.args["model_dir"]))
        
    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        if self.args['dataset'] == "cifar100":
            self.data_manager._train_trsf = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63/255),
                CIFAR10Policy(),
                transforms.ToTensor(),
            ]
        elif self.args['dataset'] == "tinyimagenet200":
            self.data_manager._train_trsf = [
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.ToPILImage()
            ]
        elif self.args['dataset'] == "imagenet100" or self.args['dataset'] == "cub200":
            self.data_manager._train_trsf = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.ToPILImage()
            ]
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        if self.args["cosine"]:
            self._network.update_fc(self._total_classes, self._cur_task)
        else:
            self._network.update_fc(self._total_classes)

        if self.al_classifier == None:
            self.al_classifier = ALClassifier(512, self._total_classes, 0, self._device,args=self.args).to(self._device)
            for name, param in self.al_classifier.named_parameters():
                param.requires_grad = False
        else:
            self.al_classifier.augment_class(data_manager.get_task_size(self._cur_task))
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        self.shot = None
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            shot=self.shot
        )
        # self.train_dataset = train_dataset
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            # unwrap to original module
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        resume = self.args['resume']  # set resume=True to use saved checkpoints
        optimizer = None
        if self._cur_task == 0:
            if resume:
                print("Loading checkpoint: {}{}_model.pth.tar".format(self.args["model_dir"], self._total_classes))
                self._network.load_state_dict(torch.load("{}{}_model.pth.tar".format(self.args["model_dir"], self._total_classes))["state_dict"], strict=False)
            self._network.to(self._device)
            if hasattr(self._network, "module"):
                self._network_module_ptr = self._network.module
            if not resume:
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=init_lr, weight_decay=init_weight_decay)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay)
                self._init_train(train_loader, test_loader, optimizer, scheduler)

            # Analytic learning phase (task 0)
            self._network.eval()
            pbar = tqdm(enumerate(train_loader), desc='Analytic Learning Phase=' + str(self._cur_task),
                             total=len(train_loader),
                             unit='batch')
            cov = torch.zeros(self.al_classifier.fe_size, self.al_classifier.fe_size).to(self._device)
            crs_cor = torch.zeros(self.al_classifier.fc.weight.size(1), self._total_classes).to(self._device)
            all_inputs = []
            all_targets = []
            with torch.no_grad():
                for i, (_, inputs, targets) in pbar:
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    all_inputs.append(inputs)
                    all_targets.append(targets)
                    out_backbone = self._network(inputs)["features"]
                    out_fe, pred = self.al_classifier(out_backbone)
                    label_onehot = F.one_hot(targets, self._total_classes).float()
                    cov += torch.t(out_fe) @ out_fe
                    crs_cor += torch.t(out_fe) @ (label_onehot)
            self.al_classifier.cov = self.al_classifier.cov + cov
            self.al_classifier.R = self.al_classifier.R + cov
            self.al_classifier.Q = self.al_classifier.Q + crs_cor
            R_inv = torch.inverse(self.al_classifier.R.cpu()).to(self._device)
            Delta = R_inv @ self.al_classifier.Q

            self.al_classifier.fc.weight = torch.nn.parameter.Parameter(
                    F.normalize(torch.t(Delta.float()), p=2, dim=-1))
            self._build_protos()
            all_inputs = torch.cat(all_inputs)
            all_targets = torch.cat(all_targets)
            fisher_backbone = compute_fisher_matrix_diag(
                args=self.args,
                model=self._network,
                device=self._device,
                optimizer=optimizer,
                x=all_inputs,
                y=all_targets,
                task_id=self._cur_task
            )
            avg_fisher = get_avg_fisher(fisher_backbone)
            print(f"Task {self._cur_task} - Average Fisher (backbone): {avg_fisher}")
            self.fisher_dict = {self._cur_task: fisher_backbone}

            # set old_task_mem for future tasks (gradient-memory of task 0)
            self.old_task_mem = compute_task_grad_memory(self._network, train_loader, self._device)

        else:
            resume = self.args['resume']
            if resume:
                print("Loading checkpoint: {}{}_model.pth.tar".format(self.args["model_dir"], self._total_classes))
                self._network.load_state_dict(torch.load("{}{}_model.pth.tar".format(self.args["model_dir"], self._total_classes))["state_dict"], strict=False)
            self._network.to(self._device)
            if hasattr(self._network, "module"):
                self._network_module_ptr = self._network.module
            if self._old_network is not None:
                self._old_network.to(self._device)

            if not resume:
                optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)

                # --- NEW: update_representation will train two branches in parallel:
                #      - self._network (normal update)
                #      - self._network_proj (projected update based on self.old_task_mem)
                # After _update_representation returns we will compute Fisher and blend the two networks by lambda_from_fisher.
                self._update_representation(train_loader, test_loader, optimizer, scheduler)

            # build protos & compute fisher on the *current* network as before
            self._build_protos()    
            all_inputs, all_targets = [], []      
            for _, inputs, targets in train_loader:
                all_inputs.append(inputs)
                all_targets.append(targets)
            all_inputs = torch.cat(all_inputs).to(self._device)
            all_targets = torch.cat(all_targets).to(self._device)      
            fisher_backbone = compute_fisher_matrix_diag(
                args=self.args,
                model=self._network,
                device=self._device,
                optimizer=optimizer,
                x=all_inputs,
                y=all_targets,
                task_id=self._cur_task          
            )
            avg_fisher = get_avg_fisher(fisher_backbone)
            print(f"Task {self._cur_task} - Average Fisher (backbone): {avg_fisher}")
            self.fisher_dict[self._cur_task] = fisher_backbone

            # compute lambda_from_fisher (giữ nguyên logic gốc)
            lambda_from_fisher = compute_fisher_merging(
                model=self._network,
                old_params=self._old_network.state_dict(),
                cur_fisher=fisher_backbone,
                old_fisher=self.fisher_dict[self._cur_task - 1]
            )
            print(f"Task {self._cur_task} - lambda_from_fisher: {lambda_from_fisher}")

            # --- NEW: Blend parameters between current network (self._network) và projected-network (self._network_proj)
            # We blend only backbone parameters (không chạm fc) giống ý đồ ban đầu.
            if self._network_proj is not None:
                state_cur = self._network.state_dict()
                state_proj = self._network_proj.state_dict()
                for name, param in self._network.named_parameters():
                    if "fc" not in name and name in state_proj:
                        blended = lambda_from_fisher * state_cur[name].to(self._device) + (1.0 - lambda_from_fisher) * state_proj[name].to(self._device)
                        param.data.copy_(blended)

            # update old_task_mem to be gradient-memory of the *blended* network for next tasks
            self.old_task_mem = compute_task_grad_memory(self._network, train_loader, self._device)

            # (Bạn vẫn giữ hoặc thay thế average_backbone_params nếu muốn — mình đã chuyển blending sang trên)

            if self.args["DPCR"]:
                print('Using DPCR')
                self._network.eval()
                self.projector = Drift_Estimator(512,False,self.args)
                self.projector.to(self._device)
                for name, param in self.projector.named_parameters():
                    param.requires_grad = False
                self.projector.eval()
                cov_pwdr = self.projector.rg_tssp * torch.eye(self.projector.fe_size).to(self._device)
                crs_cor_pwdr = torch.zeros(self.projector.fe_size, self.projector.fe_size).to(self._device)

                crs_cor_new = torch.zeros(self.al_classifier.fc.weight.size(1), self._total_classes).to(self._device)
                cov_new = torch.zeros(self.projector.fe_size, self.projector.fe_size).to(self._device)
                with torch.no_grad():
                    for i, (_, inputs, targets) in enumerate(train_loader):
                        inputs, targets = inputs.to(self._device), targets.to(self._device)
                        feats_old = self._old_network(inputs)["features"]
                        feats_new = self._network(inputs)["features"]
                        cov_pwdr += torch.t(feats_old) @ feats_old
                        cov_new += torch.t(feats_new) @ feats_new
                        crs_cor_pwdr += torch.t(feats_old) @ (feats_new)
                        label_onehot = F.one_hot(targets, self._total_classes).float()
                        crs_cor_new += torch.t(feats_new) @ (label_onehot)
                self.projector.cov = cov_pwdr
                self.projector.Q = crs_cor_pwdr
                R_inv = torch.inverse(cov_pwdr.cpu()).to(self._device)
                Delta = R_inv @ crs_cor_pwdr
                self.projector.fc.weight = torch.nn.parameter.Parameter(torch.t(Delta.float()))

                cov_prime = torch.zeros(self.al_classifier.fe_size, self.al_classifier.fe_size).to(self._device)
                Q_prime = torch.zeros(self.al_classifier.fe_size, self.al_classifier.num_classes).to(self._device)

                for class_idx in range(0, self._known_classes):
                    W = self.projector.get_weight() @ self._projectors[class_idx]
                    cov_idx = self._covs[class_idx]
                    cov_prime_idx = torch.t(W) @ cov_idx @ W
                    label = class_idx
                    label_onehot = F.one_hot(torch.tensor(label).long().to(self._device), self._total_classes).float()
                    cor_prime_idx = self.num_per_class * (torch.t(W) @ torch.t(
                        self._protos[class_idx].view(1, self.al_classifier.fe_size))) @ label_onehot.view(1, self._total_classes)
                    cov_prime += cov_prime_idx
                    Q_prime += cor_prime_idx
                    self._covs[class_idx] = cov_prime_idx
                    self._projectors[class_idx] = self.get_projector_svd(cov_prime_idx)
                    self._protos[class_idx] = self._protos[class_idx] @ W

                R_prime = cov_prime + self.al_classifier.gamma * torch.eye(self.al_classifier.fe_size).to(self._device)
                self.al_classifier.cov = cov_prime + cov_new
                self.al_classifier.Q = Q_prime + crs_cor_new
                self.al_classifier.R = R_prime+ cov_new
                R_inv = torch.inverse(self.al_classifier.R.cpu()).to(self._device)
                Delta = R_inv @ self.al_classifier.Q
                self.al_classifier.fc.weight = torch.nn.parameter.Parameter(
                        F.normalize(torch.t(Delta.float()), p=2, dim=-1))
        self._network.eval()
        test_dataset = self.data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
            )
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
        test_acc = self._compute_accuracy(self._network, test_loader)
        print(f"Task {self._cur_task} - Test Accuracy (all seen classes): {test_acc:.2f}%")

    # SVD for calculating the W_c
    def get_projector_svd(self, raw_matrix, all_non_zeros=True):
        V, S, VT = torch.svd(raw_matrix)
        if all_non_zeros:
            non_zeros_idx = torch.where(S > 0)[0]
            left_eign_vectors = V[:, non_zeros_idx]
        else:
            left_eign_vectors = V[:, :512]
        projector = left_eign_vectors @ torch.t(left_eign_vectors)
        return projector

    def _build_protos(self):
        if self.args["DPCR"]:
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx + 1),
                                                                           source='train',
                                                                           mode='test', shot=self.shot, ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)  # vectors.mean(0)
                cov = np.dot(np.transpose(vectors),vectors)
                self._protos.append(torch.tensor(class_mean).to(self._device))
                self._covs.append(torch.tensor(cov).to(self._device))
                self._projectors.append(self.get_projector_svd(self._covs[class_idx]))

    def average_backbone_params(self, lamda):
        # giữ cho tương thích cũ (không dùng trong flow mới nhưng mình vẫn để)
        old_params = {
            name: param.data.clone()
            for name, param in self._old_network.named_parameters()
            if "fc" not in name
            }

        cur_params = {
            name: param.data.clone()
            for name, param in self._network.named_parameters()
            if "fc" not in name
            }
        for name in cur_params:
            cur_params[name] = lamda * (cur_params[name]) + (1-lamda)*old_params[name]

        for name, param in self._network.named_parameters():
            if name in cur_params:
                param.data.copy_(cur_params[name])

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 25 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler, epochs_local=epochs):
        """
        Bản cập nhật representation mới:
        - Tạo 1 bản copy self._network_proj của network ban đầu (trước khi train)
        - Huấn luyện song song:
            * self._network: cập nhật bình thường (optimizer)
            * self._network_proj: cập nhật với gradient-projection (optimizer_proj)
        - Lưu self._network_proj để blending sau khi tính lambda_from_fisher
        """
        # chuẩn bị projected-network (một bản copy của network trước khi train)
        base_net = self._network.module if hasattr(self._network, "module") else self._network
        self._network_proj = base_net.copy().to(self._device)
        # đảm bảo params được trainable
        for p in self._network_proj.parameters():
            p.requires_grad = True

        # optimizer cho mạng projected
        optimizer_proj = optim.SGD(self._network_proj.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)

        prog_bar = tqdm(range(epochs_local))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            self._network_proj.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # giống trước: chỉ huấn luyện head phần mới
                fake_targets = targets - self._known_classes

                # ----- branch current (giữ nguyên behavior cũ) -----
                optimizer.zero_grad()
                logits = self._network(inputs)["logits"]
                loss_clf = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                loss = loss_clf
                loss.backward()
                optimizer.step()

                # ----- branch projected -----
                optimizer_proj.zero_grad()
                logits_p = self._network_proj(inputs)["logits"]
                loss_p = F.cross_entropy(logits_p[:, self._known_classes :], fake_targets)
                loss_p.backward()

                # lấy gradient hiện tại của projected net
                grads_cur_proj = [p.grad.detach().clone() if p.grad is not None else None for p in self._network_proj.parameters()]

                # thực hiện projection (nếu có memory)
                grads_proj = project_gradients(self._network_proj, self.old_task_mem, grads_cur_proj)

                # gán gradient đã chiếu trở lại và step
                with torch.no_grad():
                    for p, g in zip(self._network_proj.parameters(), grads_proj):
                        if p.grad is not None and g is not None:
                            p.grad.copy_(g)
                optimizer_proj.step()

                # logging (dựa trên logits của self._network)
                with torch.no_grad():
                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum().item()
                    total += len(targets)
                    losses += loss.item()

            scheduler.step()
            train_acc = np.around(100.0 * correct / total, decimals=2) if total > 0 else 0.0
            prog_bar.set_description(f"Task {_}, Epoch {epoch+1}/{epochs_local} => Loss {losses/len(train_loader):.3f}, Train_acc {train_acc:.2f}")
        logging.info("Finished update_representation")

# ======================== Helper functions ========================

def project_gradients(model, mem_grads, cur_grads):
    """
    Chiếu gradient hiện tại (cur_grads) lên không gian orthogonal của mem_grads
    mem_grads: list các gradient trung bình của các tham số từ task(s) cũ (hoặc None)
    cur_grads: list các gradient hiện tại (torch.Tensor hoặc None)
    Trả về list các gradient đã được project.
    """
    if mem_grads is None:
        # không có memory: trả lại gradient gốc
        return cur_grads

    proj_grads = []
    for g_cur, g_mem in zip(cur_grads, mem_grads):
        if g_cur is None:
            proj_grads.append(None)
            continue
        if g_mem is None:
            proj_grads.append(g_cur)
            continue

        # flatten để tính tích trong
        g_cur_v = g_cur.view(-1)
        g_mem_v = g_mem.view(-1)
        dot = torch.dot(g_cur_v, g_mem_v)
        mem_norm_sq = torch.dot(g_mem_v, g_mem_v) + 1e-12
        if dot < 0:
            # project: remove component along g_mem
            g_proj = g_cur - (dot / mem_norm_sq) * g_mem
        else:
            g_proj = g_cur
        proj_grads.append(g_proj)
    return proj_grads


def compute_task_grad_memory(model, data_loader, device):
    """
    Tính gradient trung bình của model trên toàn bộ data_loader.
    Trả về list gradients (cùng ordering với model.parameters()) để dùng làm memory cho projection.
    """
    model.eval()
    grads_mem = [torch.zeros_like(p) if p.requires_grad else None for p in model.parameters()]
    total_batches = 0
    for _, inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()
        logits = model(inputs)["logits"]
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        for idx, p in enumerate(model.parameters()):
            if p.grad is not None:
                grads_mem[idx] += p.grad.detach().clone()
        total_batches += 1

    if total_batches == 0:
        return grads_mem

    grads_mem = [ (g / total_batches) if g is not None else None for g in grads_mem ]
    return grads_mem

# -------------------------------------------------------------------------------------
# (Các hàm compute_fisher_matrix_diag, compute_fisher_merging, get_avg_fisher giữ nguyên)
def compute_fisher_matrix_diag(args, model, device, optimizer, x, y, task_id, **kwargs):
    batch_size = 128 
    # Store Fisher Information
    fisher = {n: torch.zeros(p.shape).to(device) for n, p in model.named_parameters() if p.requires_grad}
    # Do forward and backward pass to compute the fisher information
    model.train()
    r = np.arange(x.size(0))
    r = torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0, len(r), batch_size):
        if i + batch_size <= len(r):
            b = r[i : i + batch_size]
        else:
            b = r[i:]
        data = x[b].to(device)
        target = y[b].to(device)
        output = model(data)["logits"]

        if args["fisher_comp"] == "true":
            pred = output.argmax(1).flatten()
        elif args["fisher_comp"] == "empirical":
            pred = target
        else:
            raise ValueError("Unknown fisher_comp: {}".format(args["fisher_comp"]))

        loss = torch.nn.functional.cross_entropy(output, pred)
        optimizer.zero_grad()
        loss.backward()
        # Accumulate all gradients from loss with regularization
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.pow(2) * len(data)

    # Apply mean across all samples
    fisher = {n: (p / x.size(0)) for n, p in fisher.items()}
    return fisher


def compute_fisher_merging(model, old_params, cur_fisher, old_fisher):
    up = 0
    down = 0
    for n, p in model.named_parameters():
        if n in cur_fisher.keys() and "fc" not in n:
            delta = (p - old_params[n]).pow(2)
            up += torch.sum(cur_fisher[n] * delta)
            down += torch.sum((cur_fisher[n] + old_fisher[n]) * delta)

    return up / down

def get_avg_fisher(fisher):
    s = 0
    n_params = 0
    for n, p in fisher.items():
        s += torch.sum(p).item()
        n_params += p.numel()
    return s / n_params
