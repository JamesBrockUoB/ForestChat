import torch


def get_acc_seg(outputs, segs):
    outputs = torch.max(outputs, dim=1)[1]
    acc = outputs == segs
    acc = acc.view(-1)
    return acc.sum() / len(acc)


def get_acc_seg_weighted(outputs, segs):
    outputs = torch.max(outputs, dim=1)[1]
    acc = []
    for i in range(5):
        acc_temp = outputs == segs
        acc_temp = acc_temp.view(-1)
        acc.append(acc_temp.sum() / len(acc_temp))
    return torch.mean(torch.stack(acc))


def get_acc_nzero(outputs, segs):
    mask = ~segs.eq(0)
    outputs = torch.max(outputs, dim=1)[1]
    acc = torch.masked_select((outputs == segs), mask)
    return acc.sum() / len(acc)


def get_acc_class(outputs, labels):
    outputs = torch.max(outputs, dim=1)[1]
    acc = outputs == labels
    return acc.sum() / len(acc)


def get_acc_binseg(outputs, segs):
    probs = torch.sigmoid(outputs)
    preds = (probs >= 0.5).float()

    correct = (preds == segs).float()
    eps = 1e-7

    mask_1 = segs == 1
    if mask_1.any():
        acc_1 = correct[mask_1].sum() / (mask_1.sum() + eps)
    else:
        acc_1 = torch.tensor(0.0, device=outputs.device)

    mask_0 = segs == 0
    if mask_0.any():
        acc_0 = correct[mask_0].sum() / (mask_0.sum() + eps)
    else:
        acc_0 = torch.tensor(0.0, device=outputs.device)

    mean_acc = (acc_1 + acc_0) / 2

    return acc_1.item(), acc_0.item(), mean_acc.item()
