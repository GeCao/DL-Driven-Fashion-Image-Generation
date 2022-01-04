import os
import numpy as np
from src.utils import FXAA, read_img
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn

'''
file_path = os.path.join(os.path.abspath(os.curdir), 'data/transferred.png')
cloth = read_img(file_path)  # [H, W, C] BGR格式载入
cloth = FXAA(cloth)
cv2.imwrite(os.path.join(os.path.abspath(os.curdir), 'data/postprossed_transferred.png'), cloth)
'''

dir_path = os.path.join(os.path.abspath(os.curdir), 'data/dl_style_dataset/Abstraction')
all_files = os.listdir(dir_path)
widths, heights, channels = [], [], []
for i, file_name in enumerate(all_files):
    img = read_img(os.path.join(dir_path, file_name))
    widths.append(img.shape[1])
    heights.append(img.shape[0])
    channels.append(img.shape[2])
    plt.imshow(img)
    # plt.show()
widths = np.array(widths)
print("W max = {}, min = {}".format(widths.max(), widths.min()))
heights = np.array(heights)
print("H max = {}, min = {}".format(heights.max(), heights.min()))
channels = np.array(channels)
print("C max = {}, min = {}".format(channels.max(), channels.min()))
print(widths)
print(heights)