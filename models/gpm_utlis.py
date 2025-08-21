# ====== PGM Utils (copy vào đầu file hoặc 1 module utils riêng rồi import) ======
import numpy as np
import torch
import torch.nn.functional as F

def compute_conv_output_size(imgsize, kernel_size, stride=1, pad=0):
    return int(np.floor((imgsize + 2 * pad - kernel_size) / stride) + 1)

@torch.no_grad()
def get_representation_matrix_ResNet18(net, device, x, y=None):
    """
    Thu thập activation patches cho từng conv theo cách PGM.
    Trả về list các ma trận (mỗi layer 1 matrix) theo đúng thứ tự đi qua train_projected.
    """
    net.eval()
    idx = torch.randperm(x.size(0), device=device)
    b = idx[:100]  # lấy 100 mẫu
    example_data = x[b].to(device)
    _ = net(example_data)

    # Lấy activations đã được lưu sẵn trong net.act / submodules (cần có forward hooks từ IncrementalNet)
    # Bạn cần đảm bảo backbone của bạn có thuộc tính .act cho từng conv như tên dưới.
    # Nếu backbone khác, chỉnh lại danh sách dưới cho khớp.
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

    # Các thông số đặc trưng cho kiến trúc resnet-18 tùy biến của bạn (map size, stride, in_channel)
    # Hãy giữ nguyên như PGM tham chiếu; nếu kiến trúc khác, cập nhật cho khớp.
    batch_list = [10,10,10,10,10,10,10,10,50,50,50,100,100,100,100,100,100]
    stride_list = [2,1,1,1,1,2,1,1,1,2,1,1,1,2,1,1,1]
    map_list    = [84,42,42,42,42,42,21,21,21,21,11,11,11,11,6,6,6]
    in_channel  = [3,20,20,20,20,20,40,40,40,40,80,80,80,80,160,160,160]

    pad = 1
    p1d = (1,1,1,1)
    sc_list = [5,9,13]  # các vị trí có shortcut 1x1
    mat_final, mat_list, mat_sc_list = [], [], []
    for i in range(len(stride_list)):
        ksz = 3
        bsz = batch_list[i]
        st  = stride_list[i]
        k   = 0
        s = compute_conv_output_size(map_list[i], ksz, stride=st, pad=pad)
        mat = np.zeros((ksz * ksz * in_channel[i], s * s * bsz))
        act = F.pad(act_list[i], p1d, "constant", 0).detach().cpu().numpy()
        for kk in range(bsz):
            for ii in range(s):
                for jj in range(s):
                    mat[:, k] = act[kk, :, st * ii : ksz + st * ii, st * jj : ksz + st * jj].reshape(-1)
                    k += 1
        mat_list.append(mat)
        # Shortcut
        if i in sc_list:
            k = 0
            s = compute_conv_output_size(map_list[i], 1, stride=st)
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
    return mat_final

def update_GPM(model, mat_list, threshold, feature_list=None):
    """
    Trả về feature_list (list các ma trận U của từng layer).
    threshold: list (ví dụ 0.97 cho mỗi layer)
    """
    if feature_list is None:
        feature_list = []

    if len(feature_list) == 0:
        # Task đầu: lấy U theo tỉ lệ năng lượng threshold
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U, S, Vh = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S**2).sum()
            sval_ratio = (S**2) / sval_total
            r = np.sum(np.cumsum(sval_ratio) < threshold[i])
            feature_list.append(U[:, :r])
    else:
        # Task sau: trừ phần đã span bởi U trước đó rồi lấy phần dư
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()
            # projected residual
            Ui_old = feature_list[i]
            act_hat = activation - Ui_old @ (Ui_old.T @ activation)
            U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
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
                # không thêm gì
                continue

            Ui = np.hstack((feature_list[i], U[:, :r]))
            if Ui.shape[1] > Ui.shape[0]:
                feature_list[i] = Ui[:, : Ui.shape[0]]
            else:
                feature_list[i] = Ui
    return feature_list

def build_feature_projections(feature_list, device):
    """
    Từ list U (numpy) → list P = U U^T (torch, float, on device) để sử dụng khi chiếu gradient.
    """
    proj_list = []
    for U in feature_list:
        if U.size == 0:
            # layer này chưa có subspace (skip)
            proj_list.append(None)
            continue
        Ut = torch.from_numpy(U).float().to(device)  # [d, r]
        P = Ut @ Ut.t()  # [d, d]
        proj_list.append(P)
    return proj_list
