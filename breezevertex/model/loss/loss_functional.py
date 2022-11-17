import torch.nn.functional as F
import torch


def ce_loss(inputs, targets):
    """
    inputs: (torch.float32)  shape (N, C)
    targets: (torch.float32) shape (N), value {0,1,...,C-1}
    """
    # targets = targets.type(torch.int64)
    # print('targets:',targets.shape)
    # targets = torch.argmax(targets,dim=1)
    # print(targets.dtype)
    # print('inputs:',inputs)
    # print('target:',targets)

    loss = F.cross_entropy(inputs, targets)
    return loss


def mes_loss(inputs, targets):
    loss = F.mse_loss(inputs, targets)

    return loss