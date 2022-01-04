import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from .virtual_base_model import VirtualBaseModel
from ..losses.default_loss import DefaultLoss
from ..losses.vgg_loss import VGGLoss
from ..losses.style_loss import StyleLoss


class StyleTransferModel(VirtualBaseModel):
    def __init__(self, core_management):
        super(StyleTransferModel, self).__init__(core_management)

        self.cloth = None
        self.noise = None
        self.style_texture = None

        self.style_loss = None
        self.vgg_loss = None
        self.L1_loss = None

        self.total_epoch = 4000
        self.regularization = 1e-4
        self.lr = 0.005

        self.initialized = False

    def initialization(self, cloth=None, style_texture=None, noise=None):
        self.log_factory = self.core_management.log_factory
        self.cloth = cloth
        self.style_texture = style_texture
        self.noise = noise

        self.style_loss = StyleLoss()
        self.vgg_loss = VGGLoss()
        self.L1_loss = torch.nn.L1Loss()
        self.optimizer = torch.optim.Adam([self.noise], lr=self.lr)

        self.initialized = True

    def forward(self, x):
        pass

    def compute_loss(self, predicted_y, target_y):
        pass

    def train_model(self, epoch):
        self.optimizer.zero_grad()
        loss1 = self.style_loss(self.noise, self.style_texture) * 1e-1
        loss2 = self.L1_loss(self.noise, self.cloth) * 0
        loss3 = self.vgg_loss(self.noise, self.cloth) * 50
        loss = loss1 + loss2 + loss3
        if epoch % 50 == 0:
            self.log_factory.InfoLog("epoch={}/{}, style_loss={}, L1_loss={}, vgg_loss={}".format(
                epoch, self.total_epoch, loss1, loss2, loss3))

        loss.backward()
        self.optimizer.step()

    def get_noise_data(self):
        return self.noise
