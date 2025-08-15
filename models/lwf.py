# import logging
# import numpy as np
# import torch
# import os
# from collections import OrderedDict
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

# # =================== Fisher Functions from BECAME ===================
# def compute_fisher_matrix_diag(args, model, device, optimizer, x, y, task_id=None, **kwargs):
#     batch_size = 64  

#     # Bật grad và train mode để tính Fisher
#     model.train()
#     for p in model.parameters():
#         p.requires_grad_(True)

#     fisher = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}

#     r = torch.arange(x.size(0), device=device)
#     for i in range(0, len(r), batch_size):
#         b = r[i: i + batch_size]
#         data = x[b].to(device)
#         target = y[b].to(device)

#         if "space1" in kwargs.keys():
#             output_dict = model(data, space1=kwargs["space1"], space2=kwargs["space2"])
#         else:
#             output_dict = model(data)

#         if isinstance(output_dict, dict):
#             output = output_dict["logits"]
#         else:
#             output = output_dict

#         if args.get("fisher_comp", "true") == "true":
#             pred = output.argmax(1).flatten()
#         elif args.get("fisher_comp") == "empirical":
#             pred = target
#         else:
#             raise ValueError(f"Unknown fisher_comp: {args.get('fisher_comp')}")

#         loss = F.cross_entropy(output, pred)

#         optimizer.zero_grad()
#         loss.backward()  # lúc này sẽ có grad vì model đang train và requires_grad=True

#         for n, p in model.named_parameters():
#             if p.grad is not None:
#                 fisher[n] += p.grad.pow(2) * len(data)

#     fisher = {n: (p / x.size(0)) for n, p in fisher.items()}
#     return fisher

# def compute_fisher_merging(model, old_params, cur_fisher, old_fisher):
#     up = 0
#     down = 0
#     for n, p in model.named_parameters():
#         if (
#             n in cur_fisher
#             # and n in old_params
#             # and n in old_fisher
#             # and p.shape == old_params[n].shape
#         ):
#             delta = (p - old_params[n]).pow(2)
#             up += torch.sum(cur_fisher[n] * delta)
#             down += torch.sum((cur_fisher[n] + old_fisher[n]) * delta)
#         else:
#             # Bỏ qua tham số mới hoặc khác shape
#             continue
#     return up / (down + 1e-8)  # tránh chia 0


# def get_avg_fisher(fisher):
#     s = 0
#     n_params = 0
#     for _, p in fisher.items():
#         s += torch.sum(p).item()
#         n_params += p.numel()
#     return s / n_params
# # =====================================================================


# init_epoch = 2
# init_lr = 0.1
# init_milestones = [60, 120, 160]
# init_lr_decay = 0.1
# init_weight_decay = 0.0005

# # cifar100
# epochs = 2 
# lrate = 0.05
# milestones = [45, 90]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay = 2e-4
# num_workers = 8
# T = 2
# lamda = 10

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
#         self.old_params = {n: p.clone() for n, p in self._network.named_parameters() if p.requires_grad}


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
        
#         self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        
#         if self.args["cosine"]:
#             self._network.update_fc(self._total_classes, self._cur_task)
#         else:
#             self._network.update_fc(self._total_classes)

#         if self.al_classifier == None:
#             self.al_classifier = ALClassifier(512, self._total_classes, 0, self._device, args=self.args).to(self._device)
#             for name, param in self.al_classifier.named_parameters():
#                 param.requires_grad = False
#         else:
#             self.al_classifier.augment_class(data_manager.get_task_size(self._cur_task))
        
#         logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

#         self.shot = None
#         train_dataset = data_manager.get_dataset(
#             np.arange(self._known_classes, self._total_classes),
#             source="train",
#             mode="train",
#             shot=self.shot
#         )
        
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
#         # Sau khi train xong
#         cnn_acc, _ = self.eval_task()
#         print(f"[Task {self._cur_task}] Test accuracy: {cnn_acc['top1']:.2f}%")


#     # def _train(self, train_loader, test_loader):
#     #     resume = self.args['resume']
#     #     if self._cur_task == 0:
#     #         print(f"[Task {self._cur_task}] Skipping backbone training (debug mode)")
#     #         self._network.to(self._device)
#     #         if hasattr(self._network, "module"):
#     #             self._network_module_ptr = self._network.module
#     #         self._network.eval()

#     #         cov = torch.zeros(self.al_classifier.fe_size, self.al_classifier.fe_size).to(self._device)
#     #         crs_cor = torch.zeros(self.al_classifier.fc.weight.size(1), self._total_classes).to(self._device)
#     #         with torch.no_grad():
#     #             for _, inputs, targets in train_loader:
#     #                 inputs, targets = inputs.to(self._device), targets.to(self._device)
#     #                 out_backbone = self._network(inputs)["features"]
#     #                 out_fe, _ = self.al_classifier(out_backbone)
#     #                 label_onehot = F.one_hot(targets, self._total_classes).float()
#     #                 cov += torch.t(out_fe) @ out_fe
#     #                 crs_cor += torch.t(out_fe) @ label_onehot

#     #         self.al_classifier.cov = self.al_classifier.cov + cov
#     #         self.al_classifier.R = self.al_classifier.R + cov
#     #         self.al_classifier.Q = self.al_classifier.Q + crs_cor
#     #         R_inv = torch.inverse(self.al_classifier.R.cpu()).to(self._device)
#     #         Delta = R_inv @ self.al_classifier.Q
#     #         self.al_classifier.fc.weight = torch.nn.parameter.Parameter(    
#     #              F.normalize(torch.t(Delta.float()), p=2, dim=-1)
#     #          )
#     #         self._build_protos()
#     #         return  # ❌ Dừng ở đây, không train backbone task 0
#     # # Các task > 0 giữ nguyên logic cũ
#     # ...

#     def _train(self, train_loader, test_loader):
#         resume = self.args['resume']
#         if self._cur_task == 0:
#             # Task 0 như cũ...
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

#             # Lưu tham số backbone cũ để tính λ
#             old_params = {n: p.clone() for n, p in self._old_network.named_parameters()}

#             if not resume:
#                 optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)
#                 scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
#                 self._update_representation(train_loader, test_loader, optimizer, scheduler)
            
#             self._build_protos()

#             if self.args["DPCR"]:
#                 print('Using DPCR')
#                 self._network.eval()
#                 self.projector = Drift_Estimator(512, False, self.args).to(self._device)
#                 for name, param in self.projector.named_parameters():
#                     param.requires_grad = False
#                 self.projector.eval()

#                 cov_pwdr = self.projector.rg_tssp * torch.eye(self.projector.fe_size).to(self._device)
#                 crs_cor_pwdr = torch.zeros(self.projector.fe_size, self.projector.fe_size).to(self._device)
#                 crs_cor_new = torch.zeros(self.al_classifier.fc.weight.size(1), self._total_classes).to(self._device)
#                 cov_new = torch.zeros(self.projector.fe_size, self.projector.fe_size).to(self._device)

#                 with torch.no_grad():
#                     for _, inputs, targets in train_loader:
#                         inputs, targets = inputs.to(self._device), targets.to(self._device)
#                         feats_old = self._old_network(inputs)["features"]
#                         feats_new = self._network(inputs)["features"]
#                         cov_pwdr += torch.t(feats_old) @ feats_old
#                         cov_new += torch.t(feats_new) @ feats_new
#                         crs_cor_pwdr += torch.t(feats_old) @ (feats_new)
#                         label_onehot = F.one_hot(targets, self._total_classes).float()
#                         crs_cor_new += torch.t(feats_new) @ label_onehot

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

#                 # ===== Tính λ từ Fisher =====
#                 # Lấy dữ liệu cũ
#                 old_dataset = self.data_manager.get_dataset(
#                     np.arange(0, self._known_classes), source="train", mode="test", shot=None
#                 )
#                 old_loader = DataLoader(old_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#                 x_old, y_old = [], []
#                 for _, inputs, targets in old_loader:
#                     x_old.append(inputs)
#                     y_old.append(targets)
#                 x_old = torch.cat(x_old).to(self._device)
#                 y_old = torch.cat(y_old).to(self._device)

#                 # Lấy dữ liệu mới
#                 new_dataset = self.data_manager.get_dataset(
#                     np.arange(self._known_classes, self._total_classes), source="train", mode="test", shot=None
#                 )
#                 new_loader = DataLoader(new_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#                 x_new, y_new = [], []
#                 for _, inputs, targets in new_loader:
#                     x_new.append(inputs)
#                     y_new.append(targets)
#                 x_new = torch.cat(x_new).to(self._device)
#                 y_new = torch.cat(y_new).to(self._device)

#                 optimizer_dummy = torch.optim.SGD(self._network.parameters(), lr=0.001)
#                 old_fisher = compute_fisher_matrix_diag(self.args, self._old_network, self._device, optimizer_dummy, x_old, y_old, task_id=self._cur_task-1)
#                 cur_fisher = compute_fisher_matrix_diag(self.args, self._network, self._device, optimizer_dummy, x_new, y_new, task_id=self._cur_task)
#                 lamda_fisher = compute_fisher_merging(self._network, old_params, cur_fisher, old_fisher)
#                 print(f"Lambda from Fisher: {lamda_fisher.item():.4f}")

#                 # ===== RRCR có λ =====Analytic Learning Phase
#                 self.al_classifier.cov = (1-lamda_fisher) * cov_prime + lamda_fisher * cov_new
#                 self.al_classifier.Q   = (1-lamda_fisher) * Q_prime   + lamda_fisher * crs_cor_new
#                 self.al_classifier.R   = (1-lamda_fisher) * R_prime   + lamda_fisher * cov_new

#                 R_inv = torch.inverse(self.al_classifier.R).to(self._device)
#                 Delta = R_inv @ self.al_classifier.Q
#                 self.al_classifier.fc.weight = torch.nn.parameter.Parameter(
#                     F.normalize(torch.t(Delta.float()), p=2, dim=-1)
#                 )

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
#                 data, targets, idx_dataset = self.data_manager.get_dataset(
#                     np.arange(class_idx, class_idx + 1),
#                     source='train',
#                     mode='test', 
#                     shot=self.shot, 
#                     ret_data=True
#                 )
#                 idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#                 vectors, _ = self._extract_vectors(idx_loader)
#                 class_mean = np.mean(vectors, axis=0)
#                 cov = np.dot(np.transpose(vectors), vectors)
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
init_lr = 0.1
init_milestones = [60, 120, 160]
init_lr_decay = 0.1
init_weight_decay = 0.0005

# cifar100
epochs = 100
lrate = 0.05
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
        cnn_acc, _ = self.eval_task()
        print(f"[Task {self._cur_task}] Test accuracy: {cnn_acc['top1']:.2f}%")

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
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=init_lr, weight_decay=init_weight_decay)
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
                loss_kd = _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network(inputs)["logits"],
                    T,
                )

                loss = lamda * loss_kd + loss_clf

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

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
