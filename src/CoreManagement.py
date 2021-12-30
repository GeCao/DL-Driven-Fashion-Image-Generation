import os, time, math
import random
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
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
            self.transfer_style_model = StyleTransferModel(self)
            self.generate_style_model = StyleGenerationModel(self)
        else:
            self.transfer_style_model = None
            self.generate_style_model = None

        log_to_disk = param_dict['log_to_disk']
        self.log_factory = LogFactory(self, log_to_disk=log_to_disk)
        self.data_factory = DataFactory(self)
        self.content_data_loader = None
        self.style_data_loader = None

        self.train_style = None
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
        self.train_style, self.test_style = self.data_factory.get_style_dataset(train_percent=self.train_percent)

        torch.random.manual_seed(self.random_seed)

        # self.content_data_loader = DataLoader(TensorDataset(self.train_content), batch_size=self.batch_size, shuffle=True)

        self.initialized = True

    def style_generation(self):
        self.generate_style_model.initialization(self.train_style.shape)

        model_batch_size = self.generate_style_model.get_batch_size()
        model_total_epochs = self.generate_style_model.get_total_epochs()
        self.style_data_loader = DataLoader(TensorDataset(self.train_style),
                                            batch_size=model_batch_size,
                                            shuffle=True)
        self.log_factory.InfoLog("Training of style generation start")
        for epoch in range(model_total_epochs):
            for i, style_data in enumerate(self.style_data_loader):
                self.generate_style_model.train_model(epoch, i, style_data)
        self.log_factory.InfoLog("Training of style generation end")

    def style_transfer(self):
        cloth = read_img(os.path.join(self.data_path, 'cloth.jpg'))
        style_texture = read_img(os.path.join(self.data_path, 'texture.png'))
        mask = np.mean(cloth, 2)[:, :, np.newaxis] < 244
        cloth = numpy2Tensor(cloth)
        style_texture = numpy2Tensor(style_texture)
        noise = gen_noise(cloth.shape)

        noise = torch.autograd.Variable(noise, requires_grad=True)

        self.transfer_style_model.initialization(cloth, style_texture, noise)

        model_total_epochs = self.transfer_style_model.get_total_epochs()
        for epoch in range(model_total_epochs):
            self.transfer_style_model.train_model(epoch)
        noise = self.transfer_style_model.get_noise_data()
        output = tensor2numpy(noise.clamp(-1, 1)).astype(np.uint8)
        output = output * mask + (1 - mask) * 255
        cv2.imwrite(os.path.join(self.data_path, 'transferred.png'), output)

    def run(self):
        if self.model_name == 'default':
            # TODO: 1. Style generation (Ge Cao)
            # TODO: Your DataSet: self.train_style, self.test_style;    your output: style_images
            self.style_generation()

            # TODO: 2. Style Transfer (Han Yang)
            # TODO: Your DataSet: style_images, self.train_content, self.test_content(如果不需要测试集，就把它当作验证集来用)
            # self.style_transfer()

    def kill(self):
        self.log_factory.kill()
