
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


init_epoch = 10
init_lr = 0.1 
init_milestones = [60, 120, 160]
init_lr_decay = 0.1
init_weight_decay = 0.0005

# cifar100
epochs = 10 
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
                # optimizer for main branch
                optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)

                # === 1) Build feature_list from current network (GPM bases) using representation matrices ===
                # We'll collect activations from current network (eval mode)
                self._network.eval()
                # choose correct rep function depending on arch; here assume ResNet variant else alexnet
                try:
                    mat_list = get_representation_matrix_ResNet18(self._network, self._device, torch.cat([inputs for _, inputs, _ in train_loader], dim=0), None)
                except Exception:
                    # fallback: build from a small sample using stored train_loader batches
                    sample_inputs = []
                    for i, (_, inputs, _) in enumerate(train_loader):
                        sample_inputs.append(inputs)
                        if i >= 4: break
                    sample_inputs = torch.cat(sample_inputs, dim=0)
                    mat_list = get_representation_matrix_alexnet(self._network, self._device, sample_inputs, None)

                # update feature_list (numpy U matrices)
                # if we already had feature_list from previous tasks, pass them; otherwise empty to create new
                if not hasattr(self, "_feature_list") or self._feature_list is None:
                    self._feature_list = []
                self._feature_list = update_GPM(self._network, mat_list, threshold=[0.95]*len(mat_list), feature_list=self._feature_list)

                # convert feature_list -> projection matrices (torch)
                feature_proj_mats = self._build_feature_projection_matrices(self._feature_list)

                # === 2) Create projected network copy and train projected branch ===
                base_net = self._network.module if hasattr(self._network, "module") else self._network
                self._network_proj = base_net.copy().to(self._device)
                for p in self._network_proj.parameters():
                    p.requires_grad = True

                # Projected branch lr: reuse lrate or use a fraction
                lr_proj = lrate
                epochs_proj = epochs  # use same epochs variable from init scope
                self._train_projected_branch(train_loader, epochs_proj, lr_proj, feature_proj_mats)

                # === 3) Train main representation branch (existing update) ===
                # We call existing routine to update main network
                self._update_representation(train_loader, test_loader, optimizer, scheduler)
            
            lambda_from_fisher = compute_fisher_merging(
                model=self._network,
                old_params=self._old_network.state_dict(),
                cur_fisher=fisher_backbone,
                old_fisher=self.fisher_dict[self._cur_task - 1]
            )
            print(f"Task {self._cur_task} - lambda_from_fisher: {lambda_from_fisher}")

            # Blend backbone parameters between current network and projected-network (if exists)
            if hasattr(self, "_network_proj") and self._network_proj is not None:
                state_cur = self._network.state_dict()
                state_proj = self._network_proj.state_dict()
                for name, param in self._network.named_parameters():
                    if "fc" not in name and name in state_proj:
                        blended = lambda_from_fisher * state_cur[name].to(self._device) + (1.0 - lambda_from_fisher) * state_proj[name].to(self._device)
                        param.data.copy_(blended)
            else:
                # fallback: average with old network (existing behavior)
                self.average_backbone_params(lambda_from_fisher)

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
            lambda_from_fisher = compute_fisher_merging(
                model=self._network,
                old_params=self._old_network.state_dict(),
                cur_fisher=fisher_backbone,
                old_fisher=self.fisher_dict[self._cur_task - 1]
            )
            print(f"Task {self._cur_task} - lambda_from_fisher: {lambda_from_fisher}")
            self.average_backbone_params(lambda_from_fisher)
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
                        # print(feats_old)
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
    def _build_feature_projection_matrices(self, feature_list):
        """
        Convert feature_list (list of numpy U matrices per layer) into
        torch projection matrices P = U @ U.T (on device).
        Returns list of torch tensors (float) in the same order as feature_list.
        """
        proj_mats = []
        for U in feature_list:
            if U is None or U.size == 0:
                # empty -> identity of size 0 handled later
                proj_mats.append(None)
                continue
            # U: numpy array (D, r)
            P = np.dot(U, U.T)  # (D, D)
            P_t = torch.from_numpy(P).float().to(self._device)
            proj_mats.append(P_t)
        return proj_mats
    def _train_projected_branch(self, train_loader, epochs_proj, lr_proj, feature_proj_mats):
        """
        Train a copy self._network_proj with gradient projection using feature_proj_mats.
        feature_proj_mats: list of torch projection matrices (or None) mapped to conv layers in param order.
        This function updates self._network_proj in-place.
        """
        # create optimizer for projected net
        optimizer_proj = optim.SGD(self._network_proj.parameters(), lr=lr_proj, momentum=0.9, weight_decay=0.0)

        # training loop: mirror _update_representation's inner loop but with projection
        for epoch in range(epochs_proj):
            self._network_proj.train()
            losses = 0.0
            correct, total = 0, 0
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                optimizer_proj.zero_grad()
                out = self._network_proj(inputs)["logits"]
                fake_targets = targets - self._known_classes
                loss = F.cross_entropy(out[:, self._known_classes :], fake_targets)
                loss.backward()

                # Apply gradient projection layer-wise
                kk = 0
                for name, p in self._network_proj.named_parameters():
                    if p.grad is None:
                        continue
                    # apply only to conv weights (4D) and other weight matrices you want to protect
                    if p.ndim == 4:
                        P = None
                        # find corresponding proj mat
                        if kk < len(feature_proj_mats):
                            P = feature_proj_mats[kk]
                        if P is not None:
                            # p.grad shape: (out_ch, in_ch, k, k)
                            sz = p.grad.data.size(0)
                            g = p.grad.data.view(sz, -1)   # (out_ch, rest)
                            # project each row of g using P (which is rest x rest)
                            # g_proj_row = g_row - P @ g_row  if P is projection on row-space
                            # using original style: g' = g - g @ P  (keeps row dims)
                            # ensure device & dtype
                            P = P.to(g.device).type_as(g)
                            g_proj = g - torch.mm(g, P)
                            p.grad.data.copy_(g_proj.view_as(p.grad.data))
                        kk += 1
                    elif p.ndim == 1:
                        # keep bias/bn params updatable (do not zero them)
                        continue

                optimizer_proj.step()
                losses += loss.item()
                with torch.no_grad():
                    _, preds = torch.max(out, dim=1)
                    correct += preds.eq(targets).sum().item()
                    total += targets.size(0)
            # you can log per epoch if required
        # done training projected branch



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
            #cur_params[name] = old_params[name]

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

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                fake_targets = targets - self._known_classes
                loss_clf = F.cross_entropy(
                    logits[:, self._known_classes :], fake_targets
                )

                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                with torch.no_grad():
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
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)
#=====================================================================================
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



def get_representation_matrix_alexnet(net, device, x, y=None):
    # Collect activations by forward pass
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    b = r[0:125]  # Take 125 random samples
    example_data = x[b]
    example_data = example_data.to(device)
    example_out = net(example_data)

    batch_list = [2 * 12, 100, 100, 125, 125]
    mat_list = []
    act_key = list(net.act.keys())
    for i in range(len(net.map)):
        bsz = batch_list[i]
        k = 0
        if i < 3:
            ksz = net.ksize[i]
            s = compute_conv_output_size(net.map[i], net.ksize[i])
            mat = np.zeros((net.ksize[i] * net.ksize[i] * net.in_channel[i], s * s * bsz))
            act = net.act[act_key[i]].detach().cpu().numpy()
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:, k] = act[kk, :, ii : ksz + ii, jj : ksz + jj].reshape(-1)
                        k += 1
            mat_list.append(mat)
        else:
            act = net.act[act_key[i]].detach().cpu().numpy()
            activation = act[0:bsz].transpose()
            mat_list.append(activation)

    # print("-" * 30)
    # print("Representation Matrix")
    # print("-" * 30)
    # for i in range(len(mat_list)):
    #     print("Layer {} : {}".format(i + 1, mat_list[i].shape))
    # print("-" * 30)
    return mat_list


def get_representation_matrix_ResNet18(net, device, x, y=None):
    # Collect activations by forward pass
    net.eval()
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    b = r[0:100]  # ns=100 examples
    example_data = x[b]
    example_data = example_data.to(device)
    example_out = net(example_data)

    act_list = []
    act_list.extend(
        [
            net.act["conv_in"],
            net.layer1[0].act["conv_0"],
            net.layer1[0].act["conv_1"],
            net.layer1[1].act["conv_0"],
            net.layer1[1].act["conv_1"],
            net.layer2[0].act["conv_0"],
            net.layer2[0].act["conv_1"],
            net.layer2[1].act["conv_0"],
            net.layer2[1].act["conv_1"],
            net.layer3[0].act["conv_0"],
            net.layer3[0].act["conv_1"],
            net.layer3[1].act["conv_0"],
            net.layer3[1].act["conv_1"],
            net.layer4[0].act["conv_0"],
            net.layer4[0].act["conv_1"],
            net.layer4[1].act["conv_0"],
            net.layer4[1].act["conv_1"],
        ]
    )

    batch_list = [
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        50,
        50,
        50,
        100,
        100,
        100,
        100,
        100,
        100,
    ]  # scaled
    # network arch
    stride_list = [2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1]
    map_list = [84, 42, 42, 42, 42, 42, 21, 21, 21, 21, 11, 11, 11, 11, 6, 6, 6]
    in_channel = [3, 20, 20, 20, 20, 20, 40, 40, 40, 40, 80, 80, 80, 80, 160, 160, 160]

    pad = 1
    sc_list = [5, 9, 13]
    p1d = (1, 1, 1, 1)
    mat_final = []  # list containing GPM Matrices
    mat_list = []
    mat_sc_list = []
    for i in range(len(stride_list)):
        if i == 0:
            ksz = 3
        else:
            ksz = 3
        bsz = batch_list[i]
        st = stride_list[i]
        k = 0
        s = compute_conv_output_size(map_list[i], ksz, stride_list[i], pad)
        mat = np.zeros((ksz * ksz * in_channel[i], s * s * bsz))
        act = F.pad(act_list[i], p1d, "constant", 0).detach().cpu().numpy()
        for kk in range(bsz):
            for ii in range(s):
                for jj in range(s):
                    mat[:, k] = act[kk, :, st * ii : ksz + st * ii, st * jj : ksz + st * jj].reshape(-1)
                    k += 1
        mat_list.append(mat)
        # For Shortcut Connection
        if i in sc_list:
            k = 0
            s = compute_conv_output_size(map_list[i], 1, stride_list[i])
            mat = np.zeros((1 * 1 * in_channel[i], s * s * bsz))
            act = act_list[i].detach().cpu().numpy()
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:, k] = act[kk, :, st * ii : 1 + st * ii, st * jj : 1 + st * jj].reshape(-1)
                        k += 1
            mat_sc_list.append(mat)

    ik = 0
    for i in range(len(mat_list)):
        mat_final.append(mat_list[i])
        if i in [6, 10, 14]:
            mat_final.append(mat_sc_list[ik])
            ik += 1

    # print("-" * 30)
    # print("Representation Matrix")
    # print("-" * 30)
    # for i in range(len(mat_final)):
    #     print("Layer {} : {}".format(i + 1, mat_final[i].shape))
    # print("-" * 30)
    return mat_final


def update_GPM(
    model,
    mat_list,
    threshold,
    feature_list=[],
):
    # print("Threshold: ", threshold)
    if not feature_list:
        # After First Task
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U, S, Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2) / sval_total
            r = np.sum(np.cumsum(sval_ratio) < threshold[i])  # +1
            feature_list.append(U[:, 0:r])
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()
            # Projected Representation (Eq-8)
            act_hat = activation - np.dot(np.dot(feature_list[i], feature_list[i].transpose()), activation)
            U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
            # criteria (Eq-9)
            sval_hat = (S**2).sum()
            sval_ratio = (S**2) / sval_total
            accumulated_sval = (sval_total - sval_hat) / sval_total

            r = 0
            for ii in range(sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                print("Skip Updating GPM for layer: {}".format(i + 1))
                continue
            # update GPM
            Ui = np.hstack((feature_list[i], U[:, 0:r]))
            if Ui.shape[1] > Ui.shape[0]:
                feature_list[i] = Ui[:, 0 : Ui.shape[0]]
            else:
                feature_list[i] = Ui

    # print("-" * 40)
    # print("Gradient Constraints Summary")
    # print("-" * 40)
    # for i in range(len(feature_list)):
    #     print(
    #         "Layer {} : {}/{}".format(
    #             i + 1, feature_list[i].shape[1], feature_list[i].shape[0]
    #         )
    #     )
    # print("-" * 40)
    return feature_list
def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(
        np.floor(
            (Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1
        )
    )
