 ### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
from ..models.vgg19_model import Vgg19


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.weights = [1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            N, C, H, W = x_vgg[i].shape
            for n in range(N):
                phi_x = x_vgg[i][n]
                phi_y = y_vgg[i][n]
                phi_x = phi_x.reshape(C, H * W)
                phi_y = phi_y.reshape(C, H * W)
                G_x = torch.matmul(phi_x, phi_x.t())
                G_y = torch.matmul(phi_y, phi_y.t())
                loss += torch.sqrt(torch.mean((G_x - G_y) ** 2)) * self.weights[i]
        return loss
