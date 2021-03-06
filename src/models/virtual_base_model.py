import torch
import torch.nn.functional as F


class VirtualBaseModel(torch.nn.Module):
    def __init__(self, core_management):
        super(VirtualBaseModel, self).__init__()
        self.core_management = core_management
        self.log_factory = None
        self.device = core_management.device

        self.batch_size = None
        self.input_dimension = None
        self.total_epoch = 0

        self.loss = None
        self.optimizer = None

        self.initialized = False

    def initialization(self):
        self.log_factory = self.core_management.log_factory

    def get_total_epochs(self):
        return self.total_epoch

    def forward(self, input):
        pass

    def compute_loss(self, predicted_y, target_y):
        pass