import torch
import numpy as np


def compute_fisher_matrix_diag(args, model, device, optimizer, x, y, task_id, **kwargs):
    batch_size = args.batch_size_train
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
        if "space1" in kwargs.keys():  # TRGP
            output = model(data, space1=kwargs["space1"], space2=kwargs["space2"])[task_id]
        else:
            output = model(data)[task_id]

        if args.fisher_comp == "true":
            pred = output.argmax(1).flatten()
        elif args.fisher_comp == "empirical":
            pred = target
        else:
            raise ValueError("Unknown fisher_comp: {}".format(args.fisher_comp))

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
        if n in cur_fisher.keys():
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
