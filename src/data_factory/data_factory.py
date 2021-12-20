import numpy as np
import random
import torch


class DataFactory:
    def __init__(self, core_management):
        self.core_management = core_management
        self.device = self.core_management.device
        self.log_factory = self.core_management.log_factory

        self.initialized = False

    def initialization(self):
        self.initialized = True

    def get_dataset(self, train_percent):
        full_X = None  # train + test, split them later
        full_y = None  # train + test, split them later
        validation_X = None
        # TODO: please pass your read and processed dataset to core management

        # data set split as (1 - train_percent) * TestSet + train_percent * TrainSet
        idx = [i for i in range(full_X.shape[0])]
        sampled_idx = random.sample(idx, full_y.shape[0])
        indicator = np.array([False for i in range(full_X.shape[0])])
        indicator[sampled_idx[0:int(train_percent * full_X.shape[0])]] = True
        train_X = full_X[indicator == True, ...]
        train_y = full_y[indicator == True, ...]
        test_X = full_X[indicator == False, ...]
        test_y = full_y[indicator == False, ...]

        # transfer numpy data to Tensor data
        self.log_factory.InfoLog("Read data completed from X_train.csv, with shape as {}".format(train_X.shape))
        train_X = torch.autograd.Variable(torch.from_numpy(np.array(train_X)).float()).to(self.device)
        self.log_factory.InfoLog("Read data completed from y_train.csv, with shape as {}".format(train_y.shape))
        train_y = torch.autograd.Variable(torch.from_numpy(np.array(train_y)).float()).to(self.device)
        self.log_factory.InfoLog("Read data completed from X_train.csv, with shape as {}".format(test_X.shape))
        test_X = torch.autograd.Variable(torch.from_numpy(np.array(test_X)).float()).to(self.device)
        self.log_factory.InfoLog("Read data completed from y_train.csv, with shape as {}".format(test_y.shape))
        test_y = torch.autograd.Variable(torch.from_numpy(np.array(test_y)).float()).to(self.device)
        self.log_factory.InfoLog("Read data completed from X_test.csv, with shape as {}".format(validation_X.shape))

        return train_X, train_y, test_X, test_y
