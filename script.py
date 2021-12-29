import os
import numpy as np
from src.utils import FXAA, read_img
import cv2


file_path = os.path.join(os.path.abspath(os.curdir), 'data/transferred.png')
cloth = read_img(file_path)  # [W, H, C] BGR格式载入
cloth = FXAA(cloth)
cv2.imwrite(os.path.join(os.path.abspath(os.curdir), 'data/postprossed_transferred.png'), cloth)