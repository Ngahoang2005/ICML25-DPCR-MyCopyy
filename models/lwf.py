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


init_epoch = 200
init_lr = 0.001
init_milestones = [60, 120, 160]
init_lr_decay = 0.1
init_weight_decay = 0.0005

# cifar100
epochs = 100 
lrate = 0.0005
milestones = [45, 90]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2
lamda = 10

# Tiny-ImageNet200
# epochs = 100
# lrate = 0.001
# milestones = [45, 90]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay = 2e-4
# num_workers = 8
# T = 2
# lamda = 10

# imagenet100
# epochs = 100
# lrate = 0.05
# milestones = [45, 90]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay = 2e-4
# num_workers = 8
# T = 2
# lamda = 5


# fine-grained dataset
# init_lr = 0.01
# lrate = 0.005
# lamda = 20

# refer to supplementary materials for other dataset training settings

EPSILON = 1e-8

class IPTScore:
    def __init__(self, model):
        self.model = model
        self.inner_score = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        self.outer_score = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

    def calculate_score_inner(self, metric="ipt"):
        # Ở bản đơn giản nhất, inner_mask = 0.5 hết
        return {n: torch.ones_like(p) * 0.5 for n, p in self.model.named_parameters()}

    def calculate_score_outer(self, metric="ipt"):
        # Outer mask mặc định = 0.5 hết
        return {n: torch.ones_like(p) * 0.5 for n, p in self.model.named_parameters()}


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

        self._old_network = None 
        self.ipt_score = IPTScore(self._network)
        self.T = args.get("T", 2.0)
        #self.acc_per_task = []  # list lưu accuracy mỗi task
        #self.best_acc_per_task = []  # list lưu best acc đạt được tại lúc kết thúc từng task
        #self.acc_history = []

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
        if self._cur_task == 0:
            if resume:
                print("Loading checkpoint: {}{}_model.pth.tar".format(self.args["model_dir"], self._total_classes))
                self._network.load_state_dict(torch.load("{}{}_model.pth.tar".format(self.args["model_dir"], self._total_classes))["state_dict"], strict=False)
            self._network.to(self._device)
            if hasattr(self._network, "module"):
                self._network_module_ptr = self._network.module
            if not resume:
                optimizer = optim.AdamW(self._network.parameters(), lr=init_lr, weight_decay=init_weight_decay)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay)
                self._init_train(train_loader, test_loader, optimizer, scheduler)

            self._network.eval()
            pbar = tqdm(enumerate(train_loader), desc='Analytic Learning Phase=' + str(self._cur_task),
                             total=len(train_loader),
                             unit='batch')
            cov = torch.zeros(self.al_classifier.fe_size, self.al_classifier.fe_size).to(self._device)
            crs_cor = torch.zeros(self.al_classifier.fc.weight.size(1), self._total_classes).to(self._device)
            with torch.no_grad():
                for i, (_, inputs, targets) in pbar:
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
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
            #test_acc = self._compute_accuracy(self._network, test_loader)
            #self.acc_per_task.append(test_acc)
            #self.best_acc_per_task.append(test_acc)
            # forgetting = self.compute_forgetting(self._cur_task)
            # print(f"Task {self._cur_task} finished → Test Acc: {test_acc:.2f}%, Forgetting: {forgetting:.2f}%")
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
                optimizer = optim.AdamW(self._network.parameters(), lr=lrate, weight_decay=weight_decay)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
                self._update_representation(train_loader, test_loader, optimizer, scheduler)

            self._build_protos()                
                    
                    
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
                
        test_acc = self._compute_accuracy(self._network, test_loader)
        #self.acc_per_task.append(test_acc)
        #self.best_acc_per_task.append(max(self.best_acc_per_task[-1], test_acc))
        # forgetting = self.compute_forgetting(self._cur_task)
        print(f"Task {self._cur_task} finished → Test Acc: {test_acc:.2f}%")


    def update_parameters_with_task_vectors(self, theta_t, delta_in, delta_out):
        inner_mask = self.ipt_score.calculate_score_inner(metric="ipt")
        outer_mask = self.ipt_score.calculate_score_outer(metric="ipt")

        for k in inner_mask:
            inner_mask[k] = inner_mask[k].to(self._device)
            outer_mask[k] = outer_mask[k].to(self._device)

            inner = inner_mask[k]
            outer = outer_mask[k]
            assert inner.shape == outer.shape

            both_one = (inner == 1) & (outer == 1)
            inner[both_one] = 0.4
            outer[both_one] = 0.6

            both_zero = (inner == 0) & (outer == 0)
            inner[both_zero] = 0.5
            outer[both_zero] = 0.5

        with torch.no_grad():
            for name, p in self._network.named_parameters():
                updated = theta_t[name] + inner_mask[name] * delta_in[name] + outer_mask[name] * delta_out[name]
                p.copy_(updated)


    def _update_representation(self, train_loader, test_loader, optimizer, scheduler): 
        prog_bar = tqdm(range(epochs))
        for epoch in prog_bar:
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            # lưu tham số gốc theta_t
            theta_t = {name: p.clone().detach() for name, p in self._network.named_parameters()}

            data_iter = iter(train_loader)

            for cycle in range(64):  # lặp 32 lần
                # === 8 bước INNER ===
                for _ in range(4):
                    try:
                        _, inputs, targets = next(data_iter)
                    except StopIteration:
                        data_iter = iter(train_loader)
                        _, inputs, targets = next(data_iter)

                    inputs, targets = inputs.to(self._device), targets.to(self._device)

                    student_outputs = self._network(inputs)["logits"]
                    fake_targets = targets - self._known_classes
                    loss_inner = F.cross_entropy(student_outputs[:, self._known_classes:], fake_targets)

                    optimizer.zero_grad()
                    loss_inner.backward(retain_graph=True)
                    optimizer.step()

                    # delta_in
                    delta_in = {name: (p.detach() - theta_t[name]) for name, p in self._network.named_parameters()}

                    losses += loss_inner.item()
                    with torch.no_grad():
                        _, preds = torch.max(student_outputs, dim=1)
                        correct += preds.eq(targets).cpu().sum().item()
                        total += targets.size(0)

                # === 4 bước OUTER ===
                for _ in range(1):
                    try:
                        _, inputs, targets = next(data_iter)
                    except StopIteration:
                        data_iter = iter(train_loader)
                        _, inputs, targets = next(data_iter)

                    inputs, targets = inputs.to(self._device), targets.to(self._device)

                    if self._old_network is None:
                        raise RuntimeError("No teacher network for KD")
                    with torch.no_grad():
                        teacher_outputs = self._old_network(inputs)["logits"]

                    student_outputs = self._network(inputs)["logits"]
                    kd = _KD_loss(student_outputs[:, :self._known_classes], teacher_outputs, self.T)
                    fake_targets = targets - self._known_classes
                    ce_loss = F.cross_entropy(student_outputs[:, self._known_classes:], fake_targets)
                    kd_loss = 10 * kd + ce_loss
                    optimizer.zero_grad()
                    kd_loss.backward()
                    optimizer.step()

                    # delta_out
                    delta_out = {name: (p.detach() - theta_t[name]) for name, p in self._network.named_parameters()}

                    # TASK VECTOR UPDATE
                    self.update_parameters_with_task_vectors(theta_t, delta_in, delta_out)

                    losses += kd_loss.item()

            scheduler.step()
            train_acc = np.around(tensor2numpy(torch.tensor(correct)) * 100 / total, decimals=2)
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
def _KD_loss(student_logits, teacher_logits, T=2.0):
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction="batchmean"
    ) * (T * T)
# def _KD_loss(pred, soft, T):
#     pred = torch.log_softmax(pred / T, dim=1)
#     soft = torch.softmax(soft / T, dim=1)
#     return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

# import logging
# import numpy as np
# import torch
# import os
# from torch import nn
# from torch.serialization import load
# from tqdm import tqdm
# from torch import optim
# from torch.nn import functional as F
# from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
# from utils.data_manager import DummyDataset
# from utils.inc_net import IncrementalNet, CosineIncrementalNet, Drift_Estimator, ALClassifier
# from models.base import BaseLearner
# from utils.toolkit import target2onehot, tensor2numpy
# from torchvision import datasets, transforms
# from utils.autoaugment import CIFAR10Policy


# init_epoch = 200
# init_lr = 0.001
# init_milestones = [60, 120, 160]
# init_lr_decay = 0.1
# init_weight_decay = 0.0005

# # cifar100
# epochs = 100
# lrate = 0.0005
# milestones = [45, 90]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay = 2e-4
# num_workers = 8
# T = 2
# lamda = 10

# # Tiny-ImageNet200
# # epochs = 100
# # lrate = 0.001
# # milestones = [45, 90]
# # lrate_decay = 0.1
# # batch_size = 128
# # weight_decay = 2e-4
# # num_workers = 8
# # T = 2
# # lamda = 10

# # imagenet100
# # epochs = 100
# # lrate = 0.05
# # milestones = [45, 90]
# # lrate_decay = 0.1
# # batch_size = 128
# # weight_decay = 2e-4
# # num_workers = 8
# # T = 2
# # lamda = 5


# # fine-grained dataset
# # init_lr = 0.01
# # lrate = 0.005
# # lamda = 20

# # refer to supplementary materials for other dataset training settings

# EPSILON = 1e-8

# class LwF(BaseLearner):
#     def __init__(self, args):
#         super().__init__(args)
#         self.args = args
#         if self.args["dataset"] == "imagenet100" or self.args["dataset"] == "imagenet1000":
#             epochs = 100
#             lrate = 0.05
#             milestones = [45, 90]
#             lrate_decay = 0.1
#             batch_size = 128
#             weight_decay = 2e-4
#             num_workers = 8
#             T = 2
#             lamda = 5
#             self.num_per_class = 1300
#         elif self.args["dataset"] == "tinyimagenet200":
#             epochs = 100
#             lrate = 0.001
#             milestones = [45, 90]
#             lrate_decay = 0.1
#             batch_size = 128
#             weight_decay = 2e-4
#             num_workers = 8
#             T = 2
#             lamda = 10
#         print("Number of samples per class:{}".format(self.num_per_class))
#         if self.args["dataset"] == "cub200":
#             init_lr = 0.1
#             lrate = 0.05
#             lamda = 20
#             self.num_per_class = 30
#         if self.args["cosine"]:
#             self._network = CosineIncrementalNet(args, False)
#         else:
#             self._network = IncrementalNet(args, False)

#         self._protos = []
#         self.al_classifier = None
#         if self.args["DPCR"]:
#             self._covs = []
#             self._projectors = []

#     def after_task(self):
#         self._old_network = self._network.copy().freeze()
#         self._known_classes = self._total_classes
#         if not self.args['resume']:
#             if not os.path.exists(self.args["model_dir"]):
#                 os.makedirs(self.args["model_dir"])
#             self.save_checkpoint("{}".format(self.args["model_dir"]))
        

#     def incremental_train(self, data_manager):
#         self.data_manager = data_manager
#         self._cur_task += 1
#         if self.args['dataset'] == "cifar100":
#             self.data_manager._train_trsf = [
#                 transforms.RandomCrop(32, padding=4),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ColorJitter(brightness=63/255),
#                 CIFAR10Policy(),
#                 transforms.ToTensor(),
#             ]
#         elif self.args['dataset'] == "tinyimagenet200":
#             self.data_manager._train_trsf = [
#                 transforms.RandomCrop(64, padding=4),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.ToPILImage()
#             ]
#         elif self.args['dataset'] == "imagenet100" or self.args['dataset'] == "cub200":
#             self.data_manager._train_trsf = [
#                 transforms.RandomResizedCrop(224),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.ToPILImage()
#             ]
#         self._total_classes = self._known_classes + data_manager.get_task_size(
#             self._cur_task
#         )
#         if self.args["cosine"]:
#             self._network.update_fc(self._total_classes, self._cur_task)
#         else:
#             self._network.update_fc(self._total_classes)

#         if self.al_classifier == None:
#             self.al_classifier = ALClassifier(512, self._total_classes, 0, self._device,args=self.args).to(self._device)
#             for name, param in self.al_classifier.named_parameters():
#                 param.requires_grad = False
#         else:
#             self.al_classifier.augment_class(data_manager.get_task_size(self._cur_task))
#         logging.info(
#             "Learning on {}-{}".format(self._known_classes, self._total_classes)
#         )

#         self.shot = None
#         train_dataset = data_manager.get_dataset(
#             np.arange(self._known_classes, self._total_classes),
#             source="train",
#             mode="train",
#             shot=self.shot
#         )
#         # self.train_dataset = train_dataset
#         self.train_loader = DataLoader(
#             train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
#         )
#         test_dataset = data_manager.get_dataset(
#             np.arange(0, self._total_classes), source="test", mode="test"
#         )
#         self.test_loader = DataLoader(
#             test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
#         )

#         if len(self._multiple_gpus) > 1:
#             self._network = nn.DataParallel(self._network, self._multiple_gpus)
#         self._train(self.train_loader, self.test_loader)
#         if len(self._multiple_gpus) > 1:
#             self._network = self._network.module

#     def _train(self, train_loader, test_loader):
#         resume = self.args['resume']  # set resume=True to use saved checkpoints
#         if self._cur_task == 0:
#             if resume:
#                 print("Loading checkpoint: {}{}_model.pth.tar".format(self.args["model_dir"], self._total_classes))
#                 self._network.load_state_dict(torch.load("{}{}_model.pth.tar".format(self.args["model_dir"], self._total_classes))["state_dict"], strict=False)
#             self._network.to(self._device)
#             if hasattr(self._network, "module"):
#                 self._network_module_ptr = self._network.module
#             if not resume:
#                 optimizer = optim.AdamW(self._network.parameters(), lr=init_lr, weight_decay=init_weight_decay)
#                 scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay)
#                 self._init_train(train_loader, test_loader, optimizer, scheduler)

#             self._network.eval()
#             pbar = tqdm(enumerate(train_loader), desc='Analytic Learning Phase=' + str(self._cur_task),
#                              total=len(train_loader),
#                              unit='batch')
#             cov = torch.zeros(self.al_classifier.fe_size, self.al_classifier.fe_size).to(self._device)
#             crs_cor = torch.zeros(self.al_classifier.fc.weight.size(1), self._total_classes).to(self._device)
#             with torch.no_grad():
#                 for i, (_, inputs, targets) in pbar:
#                     inputs, targets = inputs.to(self._device), targets.to(self._device)
#                     out_backbone = self._network(inputs)["features"]
#                     out_fe, pred = self.al_classifier(out_backbone)
#                     label_onehot = F.one_hot(targets, self._total_classes).float()
#                     cov += torch.t(out_fe) @ out_fe
#                     crs_cor += torch.t(out_fe) @ (label_onehot)
#             self.al_classifier.cov = self.al_classifier.cov + cov
#             self.al_classifier.R = self.al_classifier.R + cov
#             self.al_classifier.Q = self.al_classifier.Q + crs_cor
#             R_inv = torch.inverse(self.al_classifier.R.cpu()).to(self._device)
#             Delta = R_inv @ self.al_classifier.Q

#             self.al_classifier.fc.weight = torch.nn.parameter.Parameter(
#                     F.normalize(torch.t(Delta.float()), p=2, dim=-1))
#             self._build_protos()
#         else:
#             resume = self.args['resume']
#             if resume:
#                 print("Loading checkpoint: {}{}_model.pth.tar".format(self.args["model_dir"], self._total_classes))
#                 self._network.load_state_dict(torch.load("{}{}_model.pth.tar".format(self.args["model_dir"], self._total_classes))["state_dict"], strict=False)
#             self._network.to(self._device)
#             if hasattr(self._network, "module"):
#                 self._network_module_ptr = self._network.module
#             if self._old_network is not None:
#                 self._old_network.to(self._device)
#             if not resume:
#                 optimizer = optim.AdamW(self._network.parameters(), lr=lrate, weight_decay=weight_decay)
#                 scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
#                 self._update_representation(train_loader, test_loader, optimizer, scheduler)
#             self._build_protos()                
                    
                    
#             if self.args["DPCR"]:
#                 print('Using DPCR')
#                 self._network.eval()
#                 self.projector = Drift_Estimator(512,False,self.args)
#                 self.projector.to(self._device)
#                 for name, param in self.projector.named_parameters():
#                     param.requires_grad = False
#                 self.projector.eval()
#                 cov_pwdr = self.projector.rg_tssp * torch.eye(self.projector.fe_size).to(self._device)
#                 crs_cor_pwdr = torch.zeros(self.projector.fe_size, self.projector.fe_size).to(self._device)

#                 crs_cor_new = torch.zeros(self.al_classifier.fc.weight.size(1), self._total_classes).to(self._device)
#                 cov_new = torch.zeros(self.projector.fe_size, self.projector.fe_size).to(self._device)
#                 with torch.no_grad():
#                     for i, (_, inputs, targets) in enumerate(train_loader):
#                         inputs, targets = inputs.to(self._device), targets.to(self._device)
#                         feats_old = self._old_network(inputs)["features"]
#                         # print(feats_old)
#                         feats_new = self._network(inputs)["features"]
#                         cov_pwdr += torch.t(feats_old) @ feats_old
#                         cov_new += torch.t(feats_new) @ feats_new
#                         crs_cor_pwdr += torch.t(feats_old) @ (feats_new)
#                         label_onehot = F.one_hot(targets, self._total_classes).float()
#                         crs_cor_new += torch.t(feats_new) @ (label_onehot)
#                 self.projector.cov = cov_pwdr
#                 self.projector.Q = crs_cor_pwdr
#                 R_inv = torch.inverse(cov_pwdr.cpu()).to(self._device)
#                 Delta = R_inv @ crs_cor_pwdr
#                 self.projector.fc.weight = torch.nn.parameter.Parameter(torch.t(Delta.float()))

#                 cov_prime = torch.zeros(self.al_classifier.fe_size, self.al_classifier.fe_size).to(self._device)
#                 Q_prime = torch.zeros(self.al_classifier.fe_size, self.al_classifier.num_classes).to(self._device)

#                 for class_idx in range(0, self._known_classes):
#                     W = self.projector.get_weight() @ self._projectors[class_idx]
#                     cov_idx = self._covs[class_idx]
#                     cov_prime_idx = torch.t(W) @ cov_idx @ W
#                     label = class_idx
#                     label_onehot = F.one_hot(torch.tensor(label).long().to(self._device), self._total_classes).float()
#                     cor_prime_idx = self.num_per_class * (torch.t(W) @ torch.t(
#                         self._protos[class_idx].view(1, self.al_classifier.fe_size))) @ label_onehot.view(1, self._total_classes)
#                     cov_prime += cov_prime_idx
#                     Q_prime += cor_prime_idx
#                     self._covs[class_idx] = cov_prime_idx
#                     self._projectors[class_idx] = self.get_projector_svd(cov_prime_idx)
#                     self._protos[class_idx] = self._protos[class_idx] @ W

#                 R_prime = cov_prime + self.al_classifier.gamma * torch.eye(self.al_classifier.fe_size).to(self._device)
#                 self.al_classifier.cov = cov_prime + cov_new
#                 self.al_classifier.Q = Q_prime + crs_cor_new
#                 self.al_classifier.R = R_prime+ cov_new
#                 R_inv = torch.inverse(self.al_classifier.R.cpu()).to(self._device)
#                 Delta = R_inv @ self.al_classifier.Q
#                 self.al_classifier.fc.weight = torch.nn.parameter.Parameter(
#                         F.normalize(torch.t(Delta.float()), p=2, dim=-1))

#         test_acc = self._compute_accuracy(self._network, test_loader)
#         print(f"Task {self._cur_task} finished → Test Acc: {test_acc:.2f}%")



#     # SVD for calculating the W_c
#     def get_projector_svd(self, raw_matrix, all_non_zeros=True):
#         V, S, VT = torch.svd(raw_matrix)
#         if all_non_zeros:
#             non_zeros_idx = torch.where(S > 0)[0]
#             left_eign_vectors = V[:, non_zeros_idx]

#         else:
#             left_eign_vectors = V[:, :512]
#         projector = left_eign_vectors @ torch.t(left_eign_vectors)
#         return projector

#     def _build_protos(self):
#         if self.args["DPCR"]:
#             for class_idx in range(self._known_classes, self._total_classes):
#                 data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx + 1),
#                                                                            source='train',
#                                                                            mode='test', shot=self.shot, ret_data=True)
#                 idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#                 vectors, _ = self._extract_vectors(idx_loader)
#                 class_mean = np.mean(vectors, axis=0)  # vectors.mean(0)
#                 cov = np.dot(np.transpose(vectors),vectors)
#                 self._protos.append(torch.tensor(class_mean).to(self._device))
#                 self._covs.append(torch.tensor(cov).to(self._device))
#                 self._projectors.append(self.get_projector_svd(self._covs[class_idx]))


#     def _init_train(self, train_loader, test_loader, optimizer, scheduler):
#         prog_bar = tqdm(range(init_epoch))
#         for _, epoch in enumerate(prog_bar):
#             self._network.train()
#             losses = 0.0
#             correct, total = 0, 0
#             for i, (_, inputs, targets) in enumerate(train_loader):
#                 inputs, targets = inputs.to(self._device), targets.to(self._device)
#                 logits = self._network(inputs)["logits"]

#                 loss = F.cross_entropy(logits, targets)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 losses += loss.item()

#                 _, preds = torch.max(logits, dim=1)
#                 correct += preds.eq(targets.expand_as(preds)).cpu().sum()
#                 total += len(targets)

#             scheduler.step()
#             train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

#             if epoch % 25 == 0:
#                 test_acc = self._compute_accuracy(self._network, test_loader)
#                 info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
#                     self._cur_task,
#                     epoch + 1,
#                     init_epoch,
#                     losses / len(train_loader),
#                     train_acc,
#                     test_acc,
#                 )
#             else:
#                 info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
#                     self._cur_task,
#                     epoch + 1,
#                     init_epoch,
#                     losses / len(train_loader),
#                     train_acc,
#                 )
#             prog_bar.set_description(info)

#         logging.info(info)

#     def _update_representation(self, train_loader, test_loader, optimizer, scheduler):

#         prog_bar = tqdm(range(epochs))
#         for _, epoch in enumerate(prog_bar):
#             self._network.train()
#             losses = 0.0
#             correct, total = 0, 0
#             for i, (_, inputs, targets) in enumerate(train_loader):
#                 inputs, targets = inputs.to(self._device), targets.to(self._device)
#                 logits = self._network(inputs)["logits"]

#                 fake_targets = targets - self._known_classes
#                 loss_clf = F.cross_entropy(
#                     logits[:, self._known_classes :], fake_targets
#                 )
#                 loss_kd = _KD_loss(
#                     logits[:, : self._known_classes],
#                     self._old_network(inputs)["logits"],
#                     T,
#                 )

#                 loss = lamda * loss_kd + loss_clf

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 losses += loss.item()

#                 with torch.no_grad():
#                     _, preds = torch.max(logits, dim=1)
#                     correct += preds.eq(targets.expand_as(preds)).cpu().sum()
#                     total += len(targets)

#             scheduler.step()
#             train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
#             if epoch % 25 == 0:
#                 test_acc = self._compute_accuracy(self._network, test_loader)
#                 info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
#                     self._cur_task,
#                     epoch + 1,
#                     epochs,
#                     losses / len(train_loader),
#                     train_acc,
#                     test_acc,
#                 )
#             else:
#                 info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
#                     self._cur_task,
#                     epoch + 1,
#                     epochs,
#                     losses / len(train_loader),
#                     train_acc,
#                 )
#             prog_bar.set_description(info)
#         logging.info(info)

# def _KD_loss(pred, soft, T):
#     pred = torch.log_softmax(pred / T, dim=1)
#     soft = torch.softmax(soft / T, dim=1)
#     return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
