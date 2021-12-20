import torch
import torch.nn as nn
from .virtual_base_model import VirtualBaseModel
from ..losses.default_loss import DefaultLoss


class StyleGenerationModel(VirtualBaseModel):
    def __init__(self, core_management):
        super(StyleGenerationModel, self).__init__(core_management)

        self.total_epoch = 2000

        self.regularization = 1e-4  # 这个值越大，最终得到得分布值越接近于恒等于mean

        self.initialized = False

    def initialization(self):
        # self.loss = R2Score(self.core_management.train_Y, self.core_management.device, self.regularization)
        self.loss = DefaultLoss(regularization=self.regularization)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0)

        self.initialized = True

    def kill(self):
        pass
