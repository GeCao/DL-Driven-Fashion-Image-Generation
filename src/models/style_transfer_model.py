import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from .virtual_base_model import VirtualBaseModel
from ..losses.default_loss import DefaultLoss


class StyleTransferModel(VirtualBaseModel):
    def __init__(self, core_management):
        super(StyleTransferModel, self).__init__(core_management)

        self.vgg_model = None
        self.vgg_feature_index = None
        self.total_epoch = 2000

        self.regularization = 1e-4  # 这个值越大，最终得到得分布值越接近于恒等于mean

        self.initialized = False

    def initialization(self):
        self.vgg_model = models.vgg16(pretrained=True).features[:]
        self.vgg_model = self.vgg_model.eval().to(self.device)
        self.vgg_feature_index = [3, 8, 15, 22]  # Relu1_2, 2_2, 3_3, 4_3

        # self.loss = R2Score(self.core_management.train_Y, self.core_management.device, self.regularization)
        self.loss = DefaultLoss(regularization=self.regularization)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0)

        self.initialized = True

    def forward(self, x):
        # TODO: implement forward of model
        return x

    def compute_loss(self, predicted_y, target_y):
        return self.loss(predicted_y, target_y)