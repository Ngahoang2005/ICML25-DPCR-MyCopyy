import numpy as np
import torch
import torch.nn.functional as F

from model import compute_conv_output_size


def train(args, model, device, x, y, optimizer, criterion, task_id):
    model.train()
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0, len(r), args.batch_size_train):
        if i + args.batch_size_train <= len(r):
            b = r[i : i + args.batch_size_train]
        else:
            b = r[i:]
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output[task_id], target)
        loss.backward()
        optimizer.step()


def train_projected(
    args,
    model,
    device,
    x,
    y,
    optimizer,
    criterion,
    task_id,
    feature_mat,
):
    model.train()
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0, len(r), args.batch_size_train):
        if i + args.batch_size_train <= len(r):
            b = r[i : i + args.batch_size_train]
        else:
            b = r[i:]
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output[task_id], target)
        loss.backward()
        # Gradient Projections
        kk = 0
        for k, (m, params) in enumerate(model.named_parameters()):
            if k < 15 and len(params.size()) != 1:
                sz = params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz, -1), feature_mat[kk]).view(
                    params.size()
                )
                kk += 1
            elif (k < 15 and len(params.size()) == 1) and task_id != 0:
                params.grad.data.fill_(0)

        optimizer.step()


def train_projected_Resnet18(args, model, device, x, y, optimizer, criterion, task_id, feature_mat):
    model.train()
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0, len(r), args.batch_size_train):
        if i + args.batch_size_train <= len(r):
            b = r[i : i + args.batch_size_train]
        else:
            b = r[i:]
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output[task_id], target)
        loss.backward()
        # Gradient Projections
        kk = 0
        for k, (m, params) in enumerate(model.named_parameters()):
            if len(params.size()) == 4:
                sz = params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz, -1), feature_mat[kk]).view(
                    params.size()
                )
                kk += 1
            elif len(params.size()) == 1 and task_id != 0:
                params.grad.data.fill_(0)

        optimizer.step()


def test(args, model, device, x, y, criterion, task_id, **kwargs):
    model.eval()
    total_loss = 0
    total_num = 0
    correct = 0
    r = np.arange(x.size(0))
    # np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)
    with torch.no_grad():
        # Loop batches
        for i in range(0, len(r), args.batch_size_test):
            if i + args.batch_size_test <= len(r):
                b = r[i : i + args.batch_size_test]
            else:
                b = r[i:]
            data = x[b]
            data, target = data.to(device), y[b].to(device)
            output = model(data)
            loss = criterion(output[task_id], target)
            pred = output[task_id].argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.item() * len(b)
            total_num += len(b)

    acc = 100.0 * correct / total_num
    final_loss = total_loss / total_num

    return final_loss, acc


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
