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

    def get_style_dataset(self, train_percent):
        full_data = []  # train + test, split them later
        # TODO: please pass your read and processed dataset to core management
        style_root_path = os.path.join(self.core_management.data_path, "dl_style_dataset")
        all_style_dir = os.listdir(style_root_path)
        for i, style_dir in enumerate(all_style_dir):
            style_dir_path = os.path.join(style_root_path, style_dir)
            curr_style_images = os.listdir(style_dir_path)
            for j, style_image in enumerate(curr_style_images):
                file_path = os.path.join(style_dir_path, style_image)
                curr_img = read_img(file_path, (64, 64))
                full_data.append(curr_img)
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
