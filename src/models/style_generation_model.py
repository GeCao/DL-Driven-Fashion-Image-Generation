import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from .virtual_base_model import VirtualBaseModel
from ..losses.default_loss import DefaultLoss
import cv2


class Generator(nn.Module):
    def __init__(self, nz, num_of_generator_features, channels):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, num_of_generator_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_of_generator_features * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(num_of_generator_features * 8, num_of_generator_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_generator_features * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(num_of_generator_features * 4, num_of_generator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_generator_features * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(num_of_generator_features * 2, num_of_generator_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_generator_features),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(num_of_generator_features, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, num_of_discriminator_features, channels):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(channels, num_of_discriminator_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(num_of_discriminator_features, num_of_discriminator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_discriminator_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(num_of_discriminator_features * 2, num_of_discriminator_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_discriminator_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(num_of_discriminator_features * 4, num_of_discriminator_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_discriminator_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(num_of_discriminator_features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class StyleGenerationModel(VirtualBaseModel):
    def __init__(self, core_management):
        super(StyleGenerationModel, self).__init__(core_management)

        self.device = None
        self.batch_size = 16
        self.total_epoch = 100
        self.lr = 0.0002
        self.beta1 = 0.5  # for Adam optimizer
        self.regularization = 1e-4  # 这个值越大，最终得到得分布值越接近于恒等于mean
        self.nz = 100  # length of latent vector
        self.num_of_generator_features = 64
        self.num_of_discriminator_features = 64
        self.channels = 3
        self.image_size = 64

        self.generator_ = None
        self.discriminator_ = None

        self.optimizer_G_ = None
        self.optimizer_D_ = None

        self.real_label = 1
        self.fake_label = 0

        self.generator_losses_ = []
        self.discriminator_losses = []

        self.initialized = False

    def initialization(self, image_shape=None):
        self.log_factory = self.core_management.log_factory
        C, H, W = image_shape[1], image_shape[2], image_shape[3]
        self.channels = C
        if H != W:
            self.log_factory.ErrorLog("An image which has different Width and Height cannot be accepted by style generation")
            exit(-1)
        self.image_size = H
        self.device = self.core_management.device
        self.generator_losses_ = []
        self.discriminator_losses = []

        self.generator_ = Generator(self.nz, self.num_of_generator_features, self.channels).to(self.device)
        self.discriminator_ = Discriminator(self.num_of_discriminator_features, self.channels).to(self.device)
        self.generator_.apply(self.weights_init)
        self.discriminator_.apply(self.weights_init)

        self.loss = torch.nn.BCELoss()
        self.optimizer_G_ = torch.optim.Adam(self.generator_.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizer_D_ = torch.optim.Adam(self.discriminator_.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.initialized = True

    def get_batch_size(self):
        return self.batch_size

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def train_model(self, epoch, i, style_data):
        self.discriminator_.zero_grad()
        label = torch.full((style_data[0].shape[0],), self.real_label, dtype=torch.float, device=self.device)
        output = self.discriminator_(style_data[0]).view(-1)
        errD_real = self.loss(output, label)
        errD_real.backward()

        noise = torch.randn(style_data[0].shape[0], self.nz, 1, 1, device=self.device)
        fake = self.generator_(noise)
        label.fill_(self.fake_label)
        output = self.discriminator_(fake.detach()).view(-1)
        errD_fake = self.loss(output, label)
        errD_fake.backward()

        errD = errD_real + errD_fake
        self.optimizer_D_.step()

        # =========== 分割线 ============

        self.generator_.zero_grad()
        label.fill_(self.real_label)
        output = self.discriminator_(fake).view(-1)
        errG = self.loss(output, label)
        errG.backward()
        self.optimizer_G_.step()

        if i % 50 == 0:
            self.log_factory.InfoLog("epoch={}/{}, i={}, loss_G={}, loss_D={}".format(
                epoch, self.get_total_epochs(), i, errG.item(), errD.item()))
        if i % 300 == 0:
            with torch.no_grad():
                noise = torch.randn(style_data[0].shape[0], self.nz, 1, 1, device=self.device)
                output_images = self.generator_(noise).permute(0, 2, 3, 1).cpu().numpy()
                for k in range(output_images.shape[0]):
                    cv2.imwrite(os.path.join(self.core_management.data_path, "generated_styles/" + str(k) + ".png"), output_images[k] * 255)


        self.generator_losses_.append(errG.item())
        self.discriminator_losses.append(errD.item())

    def kill(self):
        pass
