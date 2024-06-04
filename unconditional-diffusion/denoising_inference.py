from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import torch
from PIL import Image

generator = DiffusionPipeline.from_pretrained("./ddpm-ema-uv-spiral-128-first-300")
generator.to("cuda")
generator.scheduler = DPMSolverMultistepScheduler.from_config(generator.scheduler.config)

generator.scheduler.set_timesteps(20)

noisy_sample = torch.randn(1, 3, 128, 128)
noisy_sample = noisy_sample.to("cuda")

data = [[] for i in range(50)]



for k in range(50):
    sample = torch.randn(1, 3, 128, 128)
    sample = sample.to("cuda")
    generator.scheduler = DPMSolverMultistepScheduler.from_config(generator.scheduler.config)
    generator.scheduler.set_timesteps(20)
    img = sample.cpu().numpy()[0]
    image = np.array(img)

    u = image[0]
    v = image[1]
    data[k].append(u)

    for i, t in enumerate(tqdm(generator.scheduler.timesteps)):
        
        # 1. predict noise residual
        with torch.no_grad():
            residual = generator.unet(sample=sample, timestep=t).sample

        # 2. compute less noisy image and set x_t -> x_t-1
        sample = generator.scheduler.step(residual, t, sample).prev_sample

        img = sample.cpu().numpy()[0]
        image = np.array(img)

        u = image[0]
        v = image[1]
        data[k].append(u)

    Image.fromarray((u * 255).astype(np.uint8)).save(f"./test/{k}.png")

import pickle

with open("denoising_data.pkl", "wb") as file:
    pickle.dump(data, file)