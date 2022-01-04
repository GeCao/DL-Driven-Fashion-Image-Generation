from enum import Enum
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import cv2


class MessageAttribute(Enum):
    EInfo = 0
    EWarn = 1
    EError = 2


def model_evaluation(test_losses, train_losses, save_path, epoch_step=1):
    n = len(test_losses)
    epochs = [int(i * epoch_step) for i in range(n)]

    fig = plt.figure(1)
    plt.title("train losses-epoch")
    plt.xlabel("epoch")
    plt.plot(epochs, test_losses, 'orange', label="test")
    plt.plot(epochs, train_losses, 'b', label="train")
    plt.ylabel("loss")
    plt.legend()
    fig.savefig(save_path)


def plot_training_images(dataloader, device='cuda'):
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))


def numpy2Tensor(img):
    """
    :param img HxWxC 0~255
    :return:
    """
    img = torch.FloatTensor(img).cuda()
    img = torch.unsqueeze(img, 0)  # NHWC
    img = img.permute(0, 3, 1, 2)  # NCHW
    return (img / 255.0) * 2 - 1


def tensor2numpy(img):
    img = img.detach().cpu()
    img = (img.permute(0, 2, 3, 1) + 1) / 2 * 255
    return img[0].numpy()


def read_img(path, cut_size=(256, 256)):
    img = cv2.imread(path)
    img = cv2.resize(img, cut_size)
    return img


def gen_noise(shape):
    noise = np.zeros(shape, dtype=np.uint8)
    ### noise
    noise = cv2.randn(noise, 0, 255)
    noise = np.asarray(noise / 255, dtype=np.uint8)
    noise = torch.tensor(noise, dtype=torch.float32)
    return noise.cuda()


def FXAA(np_data, min_thresold=100, thresold=0.5):
    light = (np_data[..., 0] * 0.072 + np_data[..., 1] * 0.715 + np_data[..., 2] * 0.213) / 255
    max_lumination = np.zeros(light.shape)
    min_lumination = np.zeros(light.shape)
    light = np.pad(light, ((1, 1), (1, 1)), 'constant')
    for i in range(max_lumination.shape[0]):
        for j in range(max_lumination.shape[1]):
            max_lumination[i][j] = max(light[i + 1][j + 1], light[i + 1][j], light[i + 1][j + 2], light[i][j + 1],
                                       light[i + 2][j + 1])
            min_lumination[i][j] = min(light[i + 1][j + 1], light[i + 1][j], light[i + 1][j + 2], light[i][j + 1],
                                       light[i + 2][j + 1])
    contrast = max_lumination - min_lumination
    blend_thresold = np.where(thresold * max_lumination > min_thresold, thresold * max_lumination, min_thresold)
    need_blend = contrast > blend_thresold
    print(need_blend.shape)

    # paded_np_data = np.pad(np_data, ((1, 1), (1, 1), (0, 0)), 'constant')
    filter = np.zeros(need_blend.shape)
    for i in range(need_blend.shape[0]):
        for j in range(need_blend.shape[1]):
            if need_blend[i][j]:
                filter[i, j] = 2 * (light[i, j + 1] + light[i + 2, j + 1] + light[i + 1, j] + light[i + 1, j + 2]) + \
                               (light[i, j] + light[i, j + 2] + light[i + 2, j] + light[i + 2, j + 2])
    filter = filter / 12.0
    filter = np.abs(filter - light[1:-1, 1:-1])
    filter = np.clip(filter / contrast, 0, 1)
    filter = filter * filter
    vertical = 2 * np.abs(light[1:-1, :-2] + light[1:-1, 2:] - 2 * light[1:-1, 1:-1]) + \
               np.abs(light[:-2, :-2] + light[:-2, 2:] - 2 * light[:-2, 1:-1]) + \
               np.abs(light[2:, :-2] + light[2:, 2:] - 2 * light[2:, 1:-1])
    horizontal = 2 * np.abs(light[:-2, 1:-1] + light[2:, 1:-1] - 2 * light[1:-1, 1:-1]) + \
                 np.abs(light[:-2, :-2] + light[2:, :-2] - 2 * light[1:-1, :-2]) + \
                 np.abs(light[:-2, 2:] + light[2:, 2:] - 2 * light[1:-1, 2:])
    pixel_step = np.zeros((vertical.shape[0], vertical.shape[1], 2))
    for i in range(vertical.shape[0]):
        for j in range(vertical.shape[1]):
            if vertical[i, j] > horizontal[i, j]:
                pixel_step[i, j] = np.array([0, 1])
            else:
                pixel_step[i, j] = np.array([1, 0])
    Positive = np.abs(np.where(vertical > horizontal, light[1:-1, 2:], light[2:, 1:-1]) - light[1:-1, 1:-1])
    Negative = np.abs(np.where(vertical > horizontal, light[1:-1, :-2], light[:-2, 1:-1]) - light[1:-1, 1:-1])
    for i in range(Positive.shape[0]):
        for j in range(Positive.shape[1]):
            if Negative[i, j] > Positive[i, j]:
                pixel_step[i, j] = -pixel_step[i, j]
    output_data = np.zeros(np_data.shape)
    output_data[...] = np_data[...]
    for i in range(1, output_data.shape[0] - 1):
        for j in range(1, output_data.shape[1] - 1):
            output_data[i, j, :] = np_data[i, j, :] * (1 - filter[i, j]) + \
                                   np_data[i + int(pixel_step[i, j, 0]), j + int(pixel_step[i, j, 1]), :] * filter[i, j]
    return output_data


class Generator256(nn.Module):
    def __init__(self, nz, num_of_generator_features, channels):
        super(Generator256, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, num_of_generator_features * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_of_generator_features * 32),
            nn.ReLU(True),
            # state size. (ngf*32) x 4 x 4
            nn.ConvTranspose2d(num_of_generator_features * 32, num_of_generator_features * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_generator_features * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 8 x 8
            nn.ConvTranspose2d(num_of_generator_features * 16, num_of_generator_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_generator_features * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 16 x 16
            nn.ConvTranspose2d(num_of_generator_features * 8, num_of_generator_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_generator_features * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 32 x 32
            nn.ConvTranspose2d(num_of_generator_features * 4, num_of_generator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_generator_features * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 64 x 64
            nn.ConvTranspose2d(num_of_generator_features * 2, num_of_generator_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_generator_features),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128
            nn.ConvTranspose2d(num_of_generator_features, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 256 x 256
        )

    def forward(self, input):
        return self.main(input)


class Discriminator256(nn.Module):
    def __init__(self, num_of_discriminator_features, channels):
        super(Discriminator256, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(channels, num_of_discriminator_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(num_of_discriminator_features, num_of_discriminator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_discriminator_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(num_of_discriminator_features * 2, num_of_discriminator_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_discriminator_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(num_of_discriminator_features * 4, num_of_discriminator_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_discriminator_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(num_of_discriminator_features * 8, num_of_discriminator_features * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_discriminator_features * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(num_of_discriminator_features * 16, num_of_discriminator_features * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_discriminator_features * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(num_of_discriminator_features * 32, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class Generator128(nn.Module):
    def __init__(self, nz, num_of_generator_features, channels):
        super(Generator128, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, num_of_generator_features * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_of_generator_features * 32),
            nn.ReLU(True),
            # state size. (ngf*32) x 4 x 4
            nn.ConvTranspose2d(num_of_generator_features * 32, num_of_generator_features * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_generator_features * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 8 x 8
            nn.ConvTranspose2d(num_of_generator_features * 16, num_of_generator_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_generator_features * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 16 x 16
            nn.ConvTranspose2d(num_of_generator_features * 8, num_of_generator_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_generator_features * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 32 x 32
            nn.ConvTranspose2d(num_of_generator_features * 4, num_of_generator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_generator_features * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 64 x 64
            nn.ConvTranspose2d(num_of_generator_features * 2, num_of_generator_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_generator_features),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128
            nn.ConvTranspose2d(num_of_generator_features, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 256 x 256
        )

    def forward(self, input):
        return self.main(input)


class Discriminator128(nn.Module):
    def __init__(self, num_of_discriminator_features, channels):
        super(Discriminator128, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(channels, num_of_discriminator_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(num_of_discriminator_features, num_of_discriminator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_discriminator_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(num_of_discriminator_features * 2, num_of_discriminator_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_discriminator_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(num_of_discriminator_features * 4, num_of_discriminator_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_discriminator_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(num_of_discriminator_features * 8, num_of_discriminator_features * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_discriminator_features * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(num_of_discriminator_features * 16, num_of_discriminator_features * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_discriminator_features * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(num_of_discriminator_features * 32, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class Generator128(nn.Module):
    def __init__(self, nz, num_of_generator_features, channels):
        super(Generator128, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, num_of_generator_features * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_of_generator_features * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(num_of_generator_features * 16, num_of_generator_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_generator_features * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(num_of_generator_features * 8, num_of_generator_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_generator_features * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(num_of_generator_features * 4, num_of_generator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_generator_features * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(num_of_generator_features * 2, num_of_generator_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_generator_features),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(num_of_generator_features, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        return self.main(input)


class Discriminator128(nn.Module):
    def __init__(self, num_of_discriminator_features, channels):
        super(Discriminator128, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(channels, num_of_discriminator_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(num_of_discriminator_features, num_of_discriminator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_discriminator_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(num_of_discriminator_features * 2, num_of_discriminator_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_discriminator_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(num_of_discriminator_features * 4, num_of_discriminator_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_discriminator_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(num_of_discriminator_features * 8, num_of_discriminator_features * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_of_discriminator_features * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(num_of_discriminator_features * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
