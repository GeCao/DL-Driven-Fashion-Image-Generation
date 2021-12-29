from enum import Enum
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch
import cv2


class MessageAttribute(Enum):
    EInfo = 0
    EWarn = 1
    EError = 2


def model_evaluation(test_losses, train_losses, save_path, epoch_step=200):
    n = len(test_losses)
    epochs = [int(i * epoch_step) for i in range(n)]

    fig = plt.figure(1)
    plt.title("train losses-epoch")
    plt.xlabel("epoch")
    plt.plot(epochs, test_losses, 'orange', label="test")
    plt.plot(epochs, train_losses, 'b', label="train")
    plt.ylabel("loss")
    plt.ylim(0, 1)
    plt.legend()
    fig.save(save_path)


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


def read_img(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256))
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
