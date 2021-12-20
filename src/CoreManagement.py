import os, time, math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .data_factory.data_factory import DataFactory
from .log_factory import LogFactory
from .models.style_transfer_model import StyleTransferModel
from .models.style_generation_model import StyleGenerationModel
from .utils import *


class CoreComponent:
    def __init__(self, param_dict):
        self.root_path = os.path.abspath(os.curdir)
        self.data_path = os.path.join(self.root_path, 'data')
        print("The root path of our project: ", self.root_path)
        self.device = param_dict['device']

        self.model_name = param_dict['model']
        if self.model_name == 'default':
            self.transfer_style_model = StyleTransferModel
            self.generate_style_model = StyleGenerationModel
        else:
            self.transfer_style_model = None
            self.generate_style_model = None

        log_to_disk = param_dict['log_to_disk']
        self.log_factory = LogFactory(self, log_to_disk=log_to_disk)
        self.data_factory = DataFactory(self)
        self.content_data_loader = None
        self.style_data_loader = None

        self.train_content = None
        self.train_style = None
        self.test_content = None
        self.test_style = None

        self.train_percent = 0.95  # 如果不需要test集，就把这一参数设置成1.0
        self.batch_size = 16
        self.total_epoch = 2000
        self.random_seed = param_dict['random_seed']

        self.initialized = False

    def initialization(self):
        random.seed(self.random_seed)

        self.log_factory.initialization()
        self.log_factory.InfoLog(sentences="Log Factory fully created")

        self.data_factory.initialization()
        self.log_factory.InfoLog(sentences="Data Factory fully created")

        # TODO: DataSet Read with [B, C, H, W] mode and Process and Augmentation (Yining / Jiduan)
        # TODO:     Content DataSet: Images where their style need to be changed
        # TODO:     Style DataSet: Images which will be used to generate new style
        self.train_content, self.train_style, self.test_content, self.test_style = \
            self.data_factory.get_dataset(train_percent=self.train_percent)

        torch.random.manual_seed(self.random_seed)

        self.content_data_loader = DataLoader(TensorDataset(self.train_content), batch_size=self.batch_size, shuffle=True)
        self.style_data_loader = DataLoader(TensorDataset(self.train_style), batch_size=1, shuffle=False)
        self.generate_style_model.initialization()
        self.transfer_style_model.initialization()

        self.initialized = True

    def style_generation(self):
        pass

    def run(self):
        if self.model_name == 'default':
            # TODO: 1. Style generation (Ge Cao)
            # TODO: Your DataSet: self.train_style, self.test_style;    your output: style_images
            style_images = self.style_generation()

            # TODO: 2. Style Transfer (Han Yang)
            # TODO: Your DataSet: style_images, self.train_content, self.test_content(如果不需要测试集，就把它当作验证集来用)
            for i in range(style_images.shape[0]):
                style_image = style_images[i, ...]  # [C, H, W]
                train_loss = 0.0
                test_losses, train_losses = [], []
                for epoch in range(self.total_epoch):
                    for j, (input_train_content,) in enumerate(self.content_data_loader):
                        self.transfer_style_model.optimizer.zero_grad()
                        predicted_y = self.transfer_style_model(input_train_content)
                        train_loss = self.transfer_style_model.compute_loss(predicted_y)
                        train_loss.backward()
                        self.transfer_style_model.optimizer.step()

                    if epoch % 200 == 0:
                        with torch.no_grad():
                            test_loss = self.transfer_style_model.compute_loss(self.transfer_style_model(self.test_content),
                                                                      self.test_content.squeeze())
                            self.log_factory.InfoLog(
                                "Epoch={}, while test loss={}, train loss={}".format(epoch, test_loss, train_loss))
                            train_losses.append(train_loss.item())
                            test_losses.append(test_loss.item())

                            model_evaluation(test_losses, train_losses,
                                             save_path=os.path.join(self.data_path, 'train_eval.png'), epoch_step=200)

    def kill(self):
        self.log_factory.kill()
