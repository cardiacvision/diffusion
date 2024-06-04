# %%
import numpy as np
import PIL.Image as im
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

root = "/mnt/data1/shared/Diffusion/D-epsilon-npy-2400samples-from500sim-dt5-nozero/"

for idx, i in enumerate(tqdm(glob(f"{root}/*.npy"))):
    x = np.load(i)[:, 0]
    x[..., 1] = x[..., 1] / 3.4
    zeros = np.zeros((2400, 128, 128, 1))
    dat = (np.concatenate([x, zeros], axis=-1) * 255).astype(np.uint8)
    for idx_2, img in enumerate(dat):
        im.fromarray(img).convert("RGBA").save(f"/mnt/data_jenner/tanish/data/uv_param_data/{idx*2400 + idx_2}.png")
