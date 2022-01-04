import os
import numpy as np
import random
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from ..utils import *


class DataFactory:
    def __init__(self, core_management):
        self.core_management = core_management
        self.device = self.core_management.device
        self.log_factory = self.core_management.log_factory

        self.initialized = False

    def initialization(self):
        self.initialized = True

    def get_fractal_dataset(self, train_percent):
        full_data = []
        fractal_root_path = os.path.join(self.core_management.data_path, "fractal_dataset")
        all_fractal_dir = os.listdir(fractal_root_path)
        for i, fractal_dir in enumerate(all_fractal_dir):
            fractal_dir_path = os.path.join(fractal_root_path, fractal_dir, 'frames')
            curr_fractal_images = os.listdir(fractal_dir_path)
            curr_data = []
            for j, fractal_image in enumerate(curr_fractal_images):
                if j > 20:
                    break
                file_path = os.path.join(fractal_dir_path, fractal_image)
                curr_img = read_img(file_path, (64, 64))  # [H, W, C]
                curr_data.append(curr_img)
            curr_data = np.array(curr_data)  # [T, H, W, C]
            full_data.append(curr_data)
        full_data = np.array(full_data)  # [B, T, H, W, C]
        self.log_factory.InfoLog("full data shape = {}".format(full_data.shape))
        if full_data.shape[-1] != 3:
            self.log_factory.ErrorLog("The channel is not 3!")

        # data set split as (1 - train_percent) * TestSet + train_percent * TrainSet
        idx = [i for i in range(full_data.shape[0])]
        sampled_idx = random.sample(idx, full_data.shape[0])
        indicator = np.array([True for i in range(full_data.shape[0])])
        test_idx = [18, 146, 226, 366, 448]
        indicator[test_idx] = False
        train_data = full_data[indicator == True, ...]
        test_data = full_data[indicator == False, ...]

        # transfer numpy data to Tensor data
        train_data = torch.autograd.Variable(torch.from_numpy(np.array(train_data)).float()).cpu()
        train_data = train_data.permute(0, 2, 3, 1, 4)  # [B, H, W, T, C]
        train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], -1)  # [B, H, W, C*T]
        self.log_factory.InfoLog("Read data completed from train dataset, with shape as {}".format(train_data.shape))

        test_data = torch.autograd.Variable(torch.from_numpy(np.array(test_data)).float()).cpu()
        test_data = test_data.permute(0, 2, 3, 1, 4)  # [B, H, W, T, C]
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], -1)  # [B, H, W, C*T]
        self.log_factory.InfoLog("Read data completed from test dataset, with shape as {}".format(test_data.shape))

        return train_data, test_data

    def get_style_dataset(self, train_percent):
        full_data = []  # train + test, split them later
        style_root_path = os.path.join(self.core_management.data_path, "hard_edge_dataset/china_pattern_3811-20220102T120931Z-001")
        all_style_dir = os.listdir(style_root_path)
        for i, style_dir in enumerate(all_style_dir):
            style_dir_path = os.path.join(style_root_path, style_dir)
            curr_style_images = os.listdir(style_dir_path)
            for j, style_image in enumerate(curr_style_images):
                file_path = os.path.join(style_dir_path, style_image)
                try:
                    curr_img = read_img(file_path, (64, 64))
                    full_data.append(curr_img)
                except:
                    pass
        full_data = np.array(full_data)

        # data set split as (1 - train_percent) * TestSet + train_percent * TrainSet
        idx = [i for i in range(full_data.shape[0])]
        sampled_idx = random.sample(idx, full_data.shape[0])
        indicator = np.array([False for i in range(full_data.shape[0])])
        indicator[sampled_idx[0:int(train_percent * full_data.shape[0])]] = True
        train_data = full_data[indicator == True, ...]
        test_data = full_data[indicator == False, ...]

        # transfer numpy data to Tensor data
        train_data = torch.autograd.Variable(torch.from_numpy(np.array(train_data)).float()).to(self.device)
        train_data = train_data.permute(0, 3, 1, 2)
        self.log_factory.InfoLog("Read data completed from train dataset, with shape as {}".format(train_data.shape))

        test_data = torch.autograd.Variable(torch.from_numpy(np.array(test_data)).float()).to(self.device)
        test_data = test_data.permute(0, 3, 1, 2)
        self.log_factory.InfoLog("Read data completed from test dataset, with shape as {}".format(test_data.shape))

        return train_data, test_data
