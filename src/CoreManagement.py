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
from .models.fractal_style_model import FourierFractalModel
from .utils import *


class CoreComponent:
    def __init__(self, param_dict):
        self.root_path = os.path.abspath(os.curdir)
        self.data_path = os.path.join(self.root_path, 'data')
        print("The root path of our project: ", self.root_path)
        self.device = param_dict['device']
        self.run_type = param_dict['run_type']

        self.model_name = param_dict['model']
        if self.model_name == 'default':
            self.transfer_style_model = StyleTransferModel(self)
            self.generate_style_model = StyleGenerationModel(self)
            self.fractal_style_model = FourierFractalModel(self)
        else:
            self.transfer_style_model = None
            self.generate_style_model = None
            self.fractal_style_model = None

        log_to_disk = param_dict['log_to_disk']
        self.log_factory = LogFactory(self, log_to_disk=log_to_disk)
        self.data_factory = DataFactory(self)
        self.content_data_loader = None
        self.style_data_loader = None
        self.fractal_train_data_loader = None
        self.fractal_test_data_loader = None

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

        if self.run_type == 'fractal_generation' or 'both':
            self.train_style, self.test_style = self.data_factory.get_fractal_dataset(self.train_percent)
        elif self.run_type != 'style_transfer':
            self.train_style, self.test_style = self.data_factory.get_style_dataset(train_percent=self.train_percent)

        torch.random.manual_seed(self.random_seed)

        # self.content_data_loader = DataLoader(TensorDataset(self.train_content), batch_size=self.batch_size, shuffle=True)

        self.initialized = True

    def fractal_generation(self):
        self.fractal_style_model.initialization(self.train_style.shape)
        model_batch_size = self.fractal_style_model.get_batch_size()
        model_total_epochs = self.fractal_style_model.get_total_epochs()
        T_in = self.fractal_style_model.get_T_in()
        T = self.fractal_style_model.get_T()
        step = self.fractal_style_model.get_step()
        self.fractal_train_data_loader = DataLoader(TensorDataset(self.train_style[..., :T_in * 3],
                                                                  self.train_style[..., T_in * 3:T_in * 3 + T * 3]),
                                                    batch_size=model_batch_size,
                                                    shuffle=True)
        self.fractal_test_data_loader = DataLoader(TensorDataset(self.test_style[..., :T_in * 3],
                                                                 self.test_style[..., T_in * 3:T_in * 3 + T * 3]),
                                                   batch_size=model_batch_size,
                                                   shuffle=False)
        self.log_factory.InfoLog("Training of fractal style generation start")
        for epoch in range(model_total_epochs):
            # 1. train
            self.fractal_style_model.train_l2_step = 0
            self.fractal_style_model.train_l2_full = 0
            for i, (xx, yy) in enumerate(self.fractal_train_data_loader):
                self.fractal_style_model.train_model(epoch, i, (xx, yy))

            # 2. test
            self.fractal_style_model.test_l2_step = 0
            self.fractal_style_model.test_l2_full = 0
            self.fractal_style_model.dump_file_ptr = 0
            with torch.no_grad():
                for xx, yy in self.fractal_test_data_loader:
                    self.fractal_style_model.post_process_per_epoch(epoch, (xx, yy))

            # 3. Log
            self.fractal_style_model.scheduler.step()
            self.log_factory.InfoLog(
                "epoch={}, train_l2_step={}, train_l2_full={}, test_l2_step={}, test_l2_full={}".format(
                    epoch,
                    self.fractal_style_model.train_l2_step / self.train_style.shape[0] / (T / step),
                    self.fractal_style_model.train_l2_full / self.train_style.shape[0],
                    self.fractal_style_model.test_l2_step / self.test_style.shape[0] / (T / step),
                    self.fractal_style_model.test_l2_full / self.test_style.shape[0]))
            self.fractal_style_model.train_losses.append(self.fractal_style_model.train_l2_full / self.train_style.shape[0])
            self.fractal_style_model.test_losses.append(self.fractal_style_model.test_l2_full / self.test_style.shape[0])
        model_evaluation(self.fractal_style_model.test_losses, self.fractal_style_model.train_losses,
                         os.path.join(self.data_path, "losses.png"), 1)
        self.log_factory.InfoLog("Training of fractal style generation end")
        torch.save({'model': self.fractal_style_model.model.state_dict()},
                   os.path.join(self.data_path, 'fractal_params.pth'))
        self.log_factory.InfoLog("Save model parameters successfully")

    def fractal_use(self):
        self.fractal_style_model.initialization(self.train_style.shape)
        self.fractal_style_model.read_model_params()

        model_batch_size = self.fractal_style_model.get_batch_size()
        T_in = self.fractal_style_model.get_T_in()
        T = self.fractal_style_model.get_T()

        self.fractal_style_model.test_l2_step = 0
        self.fractal_style_model.test_l2_full = 0
        self.fractal_style_model.dump_file_ptr = 0
        self.fractal_test_data_loader = DataLoader(TensorDataset(self.test_style[..., :T_in * 3],
                                                                 self.test_style[..., T_in * 3:T_in * 3 + T * 3]),
                                                   batch_size=model_batch_size,
                                                   shuffle=False)
        with torch.no_grad():
            for xx, yy in self.fractal_test_data_loader:
                self.fractal_style_model.post_process_per_epoch(0, (xx, yy))

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
        cloth = read_img(os.path.join(self.data_path, 'content0.jpg'))
        style_texture = read_img(os.path.join(self.data_path, 'tex000.png'))

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
            if self.run_type == 'style_generation':
                self.style_generation()
            elif self.run_type == 'fractal_generation':
                self.fractal_generation()
            elif self.run_type == 'both':
                texture = self.fractal_use()

            if self.run_type == 'style_transfer' or self.run_type == 'both':
                self.style_transfer()

    def kill(self):
        self.log_factory.kill()
