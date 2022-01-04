import os, random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import LpLoss
from .virtual_base_model import VirtualBaseModel
import cv2


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(12, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


class FourierFractalModel(VirtualBaseModel):
    def __init__(self, core_management):
        super(FourierFractalModel, self).__init__(core_management)
        self.core_management = core_management
        self.data_path = self.core_management.data_path

        self.dtype = torch.float32
        self.device = 'cuda'

        self.batch_size = 8
        self.total_epoch = 500
        self.lr = 0.001
        self.scheduler_step = 100
        self.scheduler_gamma = 0.5
        self.T_in = 10  # the time for input
        self.T = 11  # the time for prediction
        self.step = 1
        self.modes = 12  # How many frequency modes do we actually expect in nnet?
        self.width = 20  # The specific width which we will lift to, by nnet

        self.optimizer = None
        self.loss = None
        self.model = None
        self.scheduler = None

        self.train_l2_step = 0
        self.train_l2_full = 0
        self.test_l2_step = 0
        self.test_l2_full = 0
        self.train_losses = []
        self.test_losses = []

        self.dump_file_ptr = 0

        self.initialized = False

    def initialization(self, image_shape=None):
        self.log_factory = self.core_management.log_factory
        self.device = self.core_management.device

        self.model = FNO2d(self.modes, self.modes, self.width).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.scheduler_step,
                                                         gamma=self.scheduler_gamma)
        self.loss = F.mse_loss  # (size_average=False)

        self.initialized = True

    def get_batch_size(self):
        return self.batch_size

    def get_T_in(self):
        return self.T_in

    def get_T(self):
        return self.T

    def get_step(self):
        return self.step

    def train_model(self, epoch, i, style_data):
        xx, yy = style_data[0].to(self.device), style_data[1].to(self.device)

        loss = 0
        for t in range(0, 3 * self.T, self.step):
            y = yy[..., t:t + self.step]
            im = self.model(xx[..., 0::3])
            loss += self.loss(im.reshape(xx.shape[0], -1), y.reshape(yy.shape[0], -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., self.step:], im), dim=-1)

        self.train_l2_step += loss.item()
        l2_full = self.loss(pred.reshape(-1), yy.reshape(-1))
        self.train_l2_full += l2_full.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def read_model_params(self):
        pth_path = os.path.join(self.data_path, 'fractal_params.pth')
        state_dict = torch.load(pth_path)
        self.model.load_state_dict(state_dict['model'])

    def post_process_per_epoch(self, epoch, input_data):
        self.test_l2_step = 0
        self.test_l2_full = 0
        xx = input_data[0].to(self.device)
        yy = input_data[1].to(self.device)
        loss = 0
        for t in range(0, self.T * 3, self.step):
            y = yy[..., t:t + self.step]
            im = self.model(xx[..., 0::3])
            loss += self.loss(im.reshape(xx.shape[0], -1), y.reshape(yy.shape[0], -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., self.step:], im), dim=-1)
        self.test_l2_step += loss.item()
        self.test_l2_full += self.loss(pred.reshape(-1), yy.reshape(-1)).item()

        # dump one of the validated files
        if epoch % (self.total_epoch // 10) == 0:
            xx = input_data[0]
            for idx in range(xx.shape[0]):
                pred_validation_x = xx[idx, ...].cpu().numpy()
                pred_validation_y = pred[idx, ...].cpu().numpy()
                real_validation_y = yy[idx, ...].cpu().numpy()
                dir_path = os.path.join(self.data_path, "fractal_results")
                if not os.path.exists(os.path.join(dir_path, 'pred_vali_{:0>4d}'.format(self.dump_file_ptr))):
                    os.mkdir(os.path.join(dir_path, 'pred_vali_{:0>4d}'.format(self.dump_file_ptr)))
                if not os.path.exists(os.path.join(dir_path, 'real_vali_{:0>4d}'.format(self.dump_file_ptr))):
                    os.mkdir(os.path.join(dir_path, 'real_vali_{:0>4d}'.format(self.dump_file_ptr)))
                pred_vali_path = os.path.join(dir_path, 'pred_vali_{:0>4d}'.format(self.dump_file_ptr))
                real_vali_path = os.path.join(dir_path, 'real_vali_{:0>4d}'.format(self.dump_file_ptr))
                for i in range(pred_validation_x.shape[2] // 3):
                    cv2.imwrite(os.path.join(pred_vali_path, '{:0>4d}.png'.format(i)), pred_validation_x[..., 3*i:3*(i+1)])
                    cv2.imwrite(os.path.join(real_vali_path, '{:0>4d}.png'.format(i)), pred_validation_x[..., 3*i:3*(i+1)])
                for i in range(pred_validation_y.shape[2] // 3):
                    cv2.imwrite(os.path.join(pred_vali_path, '{:0>4d}.png'.format(i + pred_validation_x.shape[2] // 3)),
                                pred_validation_y[..., 3*i:3*(i+1)])
                    cv2.imwrite(os.path.join(real_vali_path, '{:0>4d}.png'.format(i + pred_validation_x.shape[2] // 3)),
                                real_validation_y[..., 3*i:3*(i+1)])
                self.dump_file_ptr += 1
