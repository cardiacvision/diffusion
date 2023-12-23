import numpy as np
import matplotlib.pyplot as plt

diffusion_output = np.load("/home/tanish/Desktop/Palette-Image-to-Image-Diffusion-Models/experiments_3d_40/test_spiral_3d_230129_195724/results/test/0/results.npy")
unet_output = np.load("/home/tanish/Desktop/Unet-Baseline/unet_output_40.npy")

from skimage.transform import resize
from tqdm import tqdm
gt_large = np.load("../data/new/test_y.npy")

gt = np.zeros((5000, 64, 64, 40))

for idx, img in tqdm(enumerate(gt_large)):
    gt[idx] = resize(img, (64, 64), order=0)

diff_diffusion = gt - diffusion_output
diff_unet = gt - unet_output

layer_diffusion = ((diff_diffusion**2).sum(axis=(0,1,2))/(5000*64*64)) ** .5

layer_unet = ((diff_unet**2).sum(axis=(0,1,2))/(5000*64*64))**.5

plt.plot(range(1, 41), layer_diffusion, label="diffusion", marker="o")
plt.plot(range(1, 41), layer_unet, label="unet", marker="o")

plt.xlabel("Layer Number")
plt.ylabel("Test RMSE Per Pixel")

plt.legend()

plt.savefig("unet_diffusion_comparision_v2.png")