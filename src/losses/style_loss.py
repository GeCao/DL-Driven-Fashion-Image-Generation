import torch
import torch.nn.functional as F


class StyleLoss(torch.nn.Module):
    def __init__(self, regularization):
        super(StyleLoss, self).__init__()
        self.regularization = regularization

    def forward(self, predicted_image, content_image):
        pass