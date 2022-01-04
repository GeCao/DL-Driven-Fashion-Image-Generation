import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

root_path = os.path.dirname(os.path.dirname(os.path.abspath(os.curdir)))
root_path = os.path.join(root_path, "data/fractal_results")
dirs = os.listdir(root_path)
file_name = "0020.png"
preds = []
reals = []

for dir_ in dirs:
    if '2' in dir_:
        continue
    if 'pred' in dir_:
        preds.append(cv2.imread(os.path.join(root_path, dir_, file_name)))
    elif 'real' in dir_:
        reals.append(cv2.imread(os.path.join(root_path, dir_, file_name)))
preds = np.array(preds)
reals = np.array(reals)
# preds[..., 0], preds[..., 2] = preds[..., 2], preds[..., 0]
# reals[..., 0], reals[..., 2] = reals[..., 2], reals[..., 0]

print(np.linalg.norm(preds - reals) / (7 * 64 * 64 * 3))
print(preds.mean())

fig = plt.figure(figsize=(7, 2))
k = 0
for i in range(2):
    for j in range(7):
        plt.subplot(2, 7, k+1)
        if i == 0:
            plt.imshow(preds[j])
        elif i == 1:
            plt.imshow(reals[j])
        if j == 0:
            plt.ylabel("prediction" if i == 0 else "reference")
        plt.xticks([])
        plt.yticks([])
        k += 1
plt.show()
