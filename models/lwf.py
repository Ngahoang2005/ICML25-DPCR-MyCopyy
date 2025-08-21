
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
# ====== PGM Utils (copy vào đầu file hoặc 1 module utils riêng rồi import) ======
import numpy as np
import torch
import torch.nn.functional as F

# ---- hook để lưu activations vào model.act[...] ----
def attach_hooks(model):
    """
    Gắn forward hooks vào backbone ResNet bên trong CosineIncrementalNet.
    Lưu activation vào model.act["conv_in"], model.act["layer1_0_conv1"], ...
    """
    # model: CosineIncrementalNet
    if not hasattr(model, "convnet"):
        raise AttributeError("attach_hooks: model has no attribute 'convnet' (expected CosineIncrementalNet).")
    net = model.convnet

    model.act = {}  # dictionary lưu activations

    def get_activation(name):
        def hook(module, input, output):
            # lưu activation (detach để tránh giữ graph)
            model.act[name] = output.detach().cpu()
        return hook

    # conv1: ở convnet.conv1 là Sequential([Conv2d, BN, ReLU]) -> Conv2d nằm ở index 0
    if isinstance(net.conv1, torch.nn.Sequential):
        conv1_module = net.conv1[0]
    else:
        conv1_module = net.conv1
    conv1_module.register_forward_hook(get_activation("conv_in"))

    # Các block trong layer1..layer4, đặt key theo dạng "layer{L}_{B}_conv1"/"layer{L}_{B}_conv2"
    for layer_idx in range(1, 5):
        layer = getattr(net, f"layer{layer_idx}")
        for block_idx, block in enumerate(layer):
            # mỗi block là BasicBlock có conv1 và conv2
            block.conv1.register_forward_hook(get_activation(f"layer{layer_idx}_{block_idx}_conv1"))
            block.conv2.register_forward_hook(get_activation(f"layer{layer_idx}_{block_idx}_conv2"))


# ---- tiện ích tính kích thước conv output ----
def compute_conv_output_size(imgsize, kernel_size, stride=1, pad=0):
    return int(np.floor((imgsize + 2 * pad - kernel_size) / stride) + 1)


# ---- lấy representation matrices không dùng hard-code in_channel ----
@torch.no_grad()
def get_representation_matrix_ResNet18(model, device, x, y=None, nsamples=100):
    """
    Trả về danh sách các ma trận biểu diễn (mat_list) theo thứ tự layer conv (ResNet18 style).
    model: CosineIncrementalNet (hoặc object có .convnet = ResNet)
    x: tensor toàn bộ dữ liệu train của task (CPU or GPU)
    """
    # đảm bảo hooks đã gắn
    if not hasattr(model, "act"):
        attach_hooks(model)

    model.eval()
    # chọn nsamples ngẫu nhiên
    n_total = x.size(0)
    ns = min(nsamples, n_total)
    idx = torch.randperm(n_total)[:ns].long()
    example_data = x[idx].to(device)

    # forward qua model để hooks lưu activation (hooks lưu về CPU tensor)
    _ = model(example_data)

    # tạo danh sách keys tương ứng (theo thứ tự sẽ dùng)
    keys = ["conv_in"]
    for layer_idx in range(1, 5):
        layer = getattr(model.convnet, f"layer{layer_idx}")
        for block_idx in range(len(layer)):
            keys.append(f"layer{layer_idx}_{block_idx}_conv1")
            keys.append(f"layer{layer_idx}_{block_idx}_conv2")

    # cấu hình giống PGM tham chiếu (batch_list, stride_list)
    batch_list = [10,10,10,10,10,10,10,10,50,50,50,100,100,100,100,100,100]
    stride_list = [2,1,1,1,1,2,1,1,1,2,1,1,1,2,1,1,1]
    sc_list = [5, 9, 13]  # indices (in mat_list index space BEFORE adding sc mats) có shortcut mats (1x1)
    p = 1  # padding for patches

    mat_list = []
    mat_sc_list = []

    for i, key in enumerate(keys):
        if key not in model.act:
            raise KeyError(f"Activation for key '{key}' not found in model.act. Did you call attach_hooks() before forward?")
        act_tensor = model.act[key]  # CPU tensor shape [bsz, C, H, W]
        # chuyển numpy để build mat; dùng np.pad
        act_np = act_tensor.numpy()
        # pad spatial dims by 1
        act_np = np.pad(act_np, ((0,0),(0,0),(p,p),(p,p)), mode='constant')
        bsz = batch_list[i]
        # bảo đảm bsz <= actual samples
        if bsz > act_np.shape[0]:
            bsz = act_np.shape[0]
        C = act_np.shape[1]
        H = act_np.shape[2]
        ksz = 3
        st = stride_list[i]

        # số vị trí patch theo chiều không gian
        s = (H - ksz)//st + 1
        cols = s * s * bsz
        rows = ksz * ksz * C
        mat = np.zeros((rows, cols), dtype=np.float32)
        k = 0
        for kk in range(bsz):
            for ii in range(s):
                for jj in range(s):
                    patch = act_np[kk, :, st*ii:st*ii+ksz, st*jj:st*jj+ksz].reshape(-1)
                    mat[:, k] = patch
                    k += 1
        mat_list.append(mat)

        # nếu layer có shortcut 1x1 (theo thiết kế PGM), lưu mat_sc tương ứng
        if i in sc_list:
            # 1x1 patches
            s_sc = (H - 1)//st + 1
            mat_sc = np.zeros((1*1*C, s_sc*s_sc*bsz), dtype=np.float32)
            k2 = 0
            for kk in range(bsz):
                for ii in range(s_sc):
                    for jj in range(s_sc):
                        patch = act_np[kk, :, st*ii:st*ii+1, st*jj:st*jj+1].reshape(-1)
                        mat_sc[:, k2] = patch
                        k2 += 1
            mat_sc_list.append(mat_sc)

    # bây giờ hợp nhất mat_list và mat_sc_list theo vị trí đúng
    mat_final = []
    ik = 0
    for i in range(len(mat_list)):
        mat_final.append(mat_list[i])
        if i in [6, 10, 14]:  # theo cách PGM gốc chèn shortcut sau các layer tương ứng
            if ik < len(mat_sc_list):
                mat_final.append(mat_sc_list[ik])
                ik += 1

    return mat_final


# ---- update_GPM (giữ nguyên logic SVD) ----
def update_GPM(model, mat_list, threshold, feature_list=None):
    if feature_list is None:
        feature_list = []

    if len(feature_list) == 0:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U, S, Vh = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S**2).sum()
            sval_ratio = (S**2) / sval_total
            r = np.sum(np.cumsum(sval_ratio) < threshold[i])
            # lưu U (d, r) — d = số chiều của patch flatten
            feature_list.append(U[:, :r])
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            Ui_old = feature_list[i]
            act_hat = activation - Ui_old @ (Ui_old.T @ activation)
            U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
            sval_total = (S**2).sum()
            sval_ratio = (S**2) / sval_total
            accumulated_sval = (S**2).cumsum() / sval_total
            r = np.sum(accumulated_sval < threshold[i])
            if r > 0:
                Ui = np.hstack((Ui_old, U[:, :r]))
                feature_list[i] = Ui[:, : min(Ui.shape)]
    return feature_list
# ---- build P = U U^T để chiếu gradient ----
def build_feature_projections(feature_list, device):
    proj_list = []
    for U in feature_list:
        if U.size == 0:
            proj_list.append(None)
            continue
        Ut = torch.from_numpy(U).float().to(device)  # [d, r]
        P = Ut @ Ut.t()  # [d, d]
        proj_list.append(P)
    return proj_list


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
        self.feature_list = []    # list các U (numpy arrays) theo layer
        self.feature_mat  = []    # list các P = U U^T (torch tensors)
        self.acc_matrix   = {}    #

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
                        total=len(train_loader), unit='batch')
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

            # ==== PGM: khởi tạo feature_list từ Task 0 ====
            all_inputs = torch.cat(all_inputs).to(self._device)
            with torch.no_grad():
                if not hasattr(self._network, "act"):
                    attach_hooks(self._network)
                rep_mats = get_representation_matrix_ResNet18(self._network, self._device, all_inputs)
            thr = [0.97] * len(rep_mats)
            self.feature_list = update_GPM(self._network, rep_mats, threshold=thr, feature_list=None)
            self.feature_mat  = build_feature_projections(self.feature_list, self._device)

            # Fisher cho task 0
            all_targets = torch.cat(all_targets).to(self._device)
            fisher_backbone = compute_fisher_matrix_diag(
                args=self.args, model=self._network, device=self._device, optimizer=optimizer,
                x=all_inputs, y=all_targets, task_id=self._cur_task
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
                # Lưu init params (để tách PGM và free phases)
                init_params = {n: p.data.clone() for n, p in self._network.named_parameters() if "fc" not in n}

                # ===== Train 1: Projected training (PGM) =====
                optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
                self.train_projected(train_loader, test_loader, optimizer, scheduler)  # <== NEW
                gpm_params = {n: p.data.clone() for n, p in self._network.named_parameters() if "fc" not in n}

                # reset về init để train lần 2
                for n, p in self._network.named_parameters():
                    if n in init_params:
                        p.data.copy_(init_params[n])

                # ===== Train 2: Free update (không chiếu) =====
                optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
                self._update_representation(train_loader, test_loader, optimizer, scheduler)
                free_params = {n: p.data.clone() for n, p in self._network.named_parameters() if "fc" not in n}

                # ===== Fisher-based merging (old vs new) =====
                all_inputs, all_targets = [], []
                for _, inputs, targets in train_loader:
                    all_inputs.append(inputs)
                    all_targets.append(targets)
                all_inputs = torch.cat(all_inputs).to(self._device)
                all_targets = torch.cat(all_targets).to(self._device)

                fisher_backbone = compute_fisher_matrix_diag(
                    args=self.args, model=self._network, device=self._device, optimizer=optimizer,
                    x=all_inputs, y=all_targets, task_id=self._cur_task
                )
                self.fisher_dict[self._cur_task] = fisher_backbone
                lambda_from_fisher = compute_fisher_merging(
                    model=self._network,
                    old_params=self._old_network.state_dict(),
                    cur_fisher=fisher_backbone,
                    old_fisher=self.fisher_dict[self._cur_task - 1]
                )
                print(f"Task {self._cur_task} - lambda_from_fisher: {lambda_from_fisher}")

                # Merge giữa 2 bộ tham số (GPM vs Free)
                for n, p in self._network.named_parameters():
                    if n in gpm_params:
                        merged = lambda_from_fisher * free_params[n] + (1 - lambda_from_fisher) * gpm_params[n]
                        p.data.copy_(merged)

                # ====== Cập nhật GPM sau khi hoàn thành task (từ dữ liệu hiện tại) ======
                with torch.no_grad():
                    if not hasattr(self._network, "act"):
                        attach_hooks(self._network)
                    rep_mats = get_representation_matrix_ResNet18(self._network, self._device, all_inputs)
                thr = [0.97] * len(rep_mats)
                self.feature_list = update_GPM(self._network, rep_mats, threshold=thr, feature_list=self.feature_list)
                self.feature_mat  = build_feature_projections(self.feature_list, self._device)

            # DPCR như code cũ của bạn (giữ nguyên)
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

        # ====== Đánh giá tổng thể + lưu acc theo từng task để tính forgetting ======
        self._network.eval()
        test_dataset = self.data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
        test_acc = self._compute_accuracy(self._network, test_loader)
        print(f"Task {self._cur_task} - Test Accuracy (all seen classes): {test_acc:.2f}%")

        # Acc riêng từng task để tính Forgetting
        acc_per_task = []
        task_size_fn = self.data_manager.get_task_size
        for prev_task in range(self._cur_task + 1):
            start_c = sum(task_size_fn(t) for t in range(prev_task))
            end_c   = start_c + task_size_fn(prev_task)
            prev_dataset = self.data_manager.get_dataset(
                np.arange(start_c, end_c), source="test", mode="test"
            )
            prev_loader = DataLoader(prev_dataset, batch_size=128, shuffle=False, num_workers=4)
            acc_prev = self._compute_accuracy(self._network, prev_loader)
            acc_per_task.append(acc_prev)
            print(f"   Accuracy on Task {prev_task}: {acc_prev:.2f}%")
        self.acc_matrix[self._cur_task] = acc_per_task

        if self._cur_task > 0:
            # Forgetting score: mean over old tasks of (max previous acc - current acc)
            forgetting = []
            for i in range(self._cur_task):  # từng task cũ
                acc_hist = [self.acc_matrix[k][i] for k in range(i, self._cur_task + 1)]  # từ khi xong task i đến hiện tại
                max_prev = max(acc_hist[:-1])  # peak trước hiện tại
                last_acc = acc_hist[-1]
                forgetting.append(max_prev - last_acc)
            avg_forgetting = float(np.mean(forgetting)) if len(forgetting) else 0.0
            print(f"Task {self._cur_task} - Forgetting Score: {avg_forgetting:.2f}")

    # ----------------- NEW: train_projected (PGM) -----------------
    def train_projected(self, train_loader, test_loader, optimizer, scheduler):
        """
        Train với Gradient Projection Method:
        - Xây subspace (self.feature_list) → self.feature_mat (P = UU^T)
        - Chỉ chiếu gradient của các conv weights (len==4)
        """
        device = self._device
        criterion = torch.nn.CrossEntropyLoss()

        # Bảo đảm đã có feature_mat (nếu mới vào task>0 mà vẫn trống, xây từ dữ liệu hiện tại)
        if len(self.feature_list) == 0 or len(self.feature_mat) == 0:
            all_inputs = []
            for _, inputs, _ in train_loader:
                all_inputs.append(inputs)
            all_inputs = torch.cat(all_inputs).to(device)
            with torch.no_grad():
                if not hasattr(self._network, "act"):
                    attach_hooks(self._network)
                rep_mats = get_representation_matrix_ResNet18(self._network, device, all_inputs)
            thr = [0.97] * len(rep_mats)
            self.feature_list = update_GPM(self._network, rep_mats, threshold=thr, feature_list=None)
            self.feature_mat  = build_feature_projections(self.feature_list, device)

        self._network.train()
        for epoch in range(epochs):
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                out = self._network(inputs)["logits"]
                # chỉ tối ưu các class mới
                fake_targets = targets - self._known_classes
                loss = criterion(out[:, self._known_classes:], fake_targets)
                loss.backward()

                # Gradient Projections — theo PGM tham chiếu:
                kk = 0
                for name, params in self._network.named_parameters():
                    if params.grad is None:
                        continue
                    if params.dim() == 4:  # conv weight: (out_ch, in_ch, k, k)
                        grad_vec = params.grad.data.view(-1, 1)   # [N,1]
                        P = self.feature_mat[kk]                  # [N,N]
                        if P is not None and P.shape[0] == grad_vec.shape[0]:
                            grad_proj = grad_vec - P @ (P @ grad_vec)  # g' = g - PPg
                            params.grad.data.copy_(grad_proj.view_as(params.grad.data))
                        kk += 1

                    elif params.dim() == 1 and self._cur_task != 0:
                    # zero biases for non-first task
                        params.grad.data.zero_()

                optimizer.step()
            scheduler.step()
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

        self._protos = []
        self._projectors = []

        for class_idx in range(len(self._covs)):
            projector = self.get_projector_svd(self._covs[class_idx])
            self._projectors.append(projector)

        # Tính prototype vector cho class_idx
        # (trung bình các feature vectors trong class đó)
            feats, labels = [], []
            loader = self.data_manager.get_dataset(
            np.arange(class_idx, class_idx + 1), source="train", mode="test"
        )
            loader = DataLoader(loader, batch_size=128, shuffle=False, num_workers=4)

            with torch.no_grad():
                for _, inputs, targets in loader:
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    out = self._network(inputs)["features"]
                    mask = targets == class_idx
                    if mask.sum() > 0:
                            feats.append(out[mask])
                    labels.append(targets[mask])

            if len(feats) > 0:
                feats = torch.cat(feats, dim=0)
                proto = feats.mean(0, keepdim=True)  # mean feature = prototype
                self._protos.append(proto)
            else:
            # nếu không có sample thì thêm vector zero
                self._protos.append(torch.zeros(1, self.al_classifier.fe_size).to(self._device))


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
# init_lr = 0.1
# init_milestones = [60, 120, 160]
# init_lr_decay = 0.1
# init_weight_decay = 0.0005

# # cifar100
# epochs = 100
# lrate = 0.05
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
#                 optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=init_lr, weight_decay=init_weight_decay)
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
#                 optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)
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
#         self._network.eval()
#         test_dataset = self.data_manager.get_dataset(
#             np.arange(0, self._total_classes), source="test", mode="test"
#             )
#         test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
#         test_acc = self._compute_accuracy(self._network, test_loader)
#         print(f"Task {self._cur_task} - Test Accuracy (all seen classes): {test_acc:.2f}%")

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

