import torch
import torch.nn.functional as F


class ContentLoss(torch.nn.Module):
    def __init__(self, regularization):
        super(ContentLoss, self).__init__()
        self.regularization = regularization

    def forward(self, predicted_image, content_image):
        return torch.norm(predicted_image - content_image, p=2)