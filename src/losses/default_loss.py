import torch
import torch.nn.functional as F


class DefaultLoss(torch.nn.Module):
    def __init__(self, regularization):
        super(DefaultLoss, self).__init__()
        self.regularization = regularization

    def forward(self, predicted_image, content_image):
        pass
