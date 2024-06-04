#%%
import numpy as np

params = np.load("/home/tanish/Desktop/Palette-Image-to-Image-Diffusion-Models/experiments_cond/test_cond_period_range_240520_104242/results/test/0/params.npy")
gens = np.load("/home/tanish/Desktop/Palette-Image-to-Image-Diffusion-Models/experiments_cond/test_cond_period_range_240520_104242/results/test/0/results.npy")

# %%
param_1 = [0.00025, 0.0005, 0.001, 0.00175, 0.0275]
param_2 = [0.0035, 0.0285, 0.0535, 0.0785, 0.1035]
label_matrix = -np.log(np.dstack(np.meshgrid(param_1, param_2)))
# %%
data = {}
reverse = {}
for i in range(5):
    for j in range(5):
        data[f"{j+1}-{i+1}"] = label_matrix[i, j]
        reverse[tuple(label_matrix[i, j])] = f"{j+1}-{i+1}"
        d = gens[(params[:, 0] == label_matrix[i, j][0]) & (params[:, 1] == label_matrix[i, j][1])]
        np.save(f"./params_generated_npys/{j+1}-{i+1}.npy", d)
# %%
d.shape
# %%
label_matrix[i, j]
# %%
