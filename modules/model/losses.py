"""Module with custom losses"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):

    def __init__(self, num_classes: int, smoothing: float, dim=-1):
        super().__init__()

        assert 0 <= smoothing < 1
        assert num_classes >= 1

        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.dim = dim

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # It applies softmax first, so you need to pass raw values wuth softmax or sigmoid
        pred = F.log_softmax(pred, dim=self.dim)

        with torch.no_grad():
            true_dist = torch.zeros_like(input=pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))

            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        result = torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

        return result


if __name__ == '__main__':
    pred = torch.tensor([[-0.15, 1.48, -1.78, 1.15]])
    target = torch.tensor([1])

    loss1 = LabelSmoothingLoss(num_classes=4, smoothing=0.2)

    res1 = loss1(pred, target)

    print(f'Result1: {res1}')
