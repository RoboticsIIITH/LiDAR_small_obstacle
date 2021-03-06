import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target, ignore_index=-100):

        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C

        target = target.view(-1, 1)
        target = target.long()

        ignore_mask = (target != ignore_index).squeeze()
        target = target[ignore_mask]
        input = input[ignore_mask]

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class MSELoss(nn.Module):

    def __init__(self,weights=None):
        super(MSELoss,self).__init__()
        self.weights = weights

    def forward(self,input,target,ignore_index=-100):
        if input.dim()>2:
            input = input.transpose(1, 2)                         # N,C,L => N,L,C
            input = input.contiguous().view(-1, input.size(2))    # N,L,C => N*L,C
        target = target.view(-1, 1)

        ignore_mask = (target != ignore_index).squeeze()
        target = target[ignore_mask].squeeze()
        input = input[ignore_mask].squeeze()

        if self.weights is not None:
            if self.weights.type() != input.data.type():
                self.weights = self.weights.type_as(input.data)
            at = self.weights.gather(0, target.data.view(-1).long())

        loss = (target - input)**2
        print(loss[230:240].data)
        loss = (at*loss).mean()
        return loss.mean()
