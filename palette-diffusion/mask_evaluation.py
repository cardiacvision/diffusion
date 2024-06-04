import numpy as np
import matplotlib.pyplot as plt

ax1 = plt.subplot()

gts = np.load("experiments_2d_temporal/test_inpainting_2d_simple_230808_220813/results/test/0/gts.npy") / 255
preds = np.load("experiments_2d_temporal/test_inpainting_2d_simple_230808_220813/results/test/0/results.npy") / 255

mask_range = np.arange(0.05, 0.81, 0.05)

# rmse = np.abs((gts - preds)).sum(axis=(1, 2))
rmse = ((gts - preds) ** 2).sum(axis=(1, 2))



rmse = rmse.reshape(-1, 200) / (mask_range * (128*128)).repeat(200).reshape(-1, 200)
rmse = rmse.reshape(16, -1).sum(axis=1) / 200
rmse = rmse ** .5

ax1.plot(mask_range, rmse, marker="o", color="black")

gts = np.load("experiments_2d_temporal/test_inpainting_2d_temporal_mask_range_230623_213422/results/test/0/gts.npy") / 255
preds = np.load("experiments_2d_temporal/test_inpainting_2d_temporal_mask_range_230623_213422/results/test/0/results.npy") / 255

mask_range = np.arange(0.05, 0.81, 0.05)

# rmse = np.abs((gts - preds)).sum(axis=(1, 2))
rmse = ((gts - preds) ** 2).sum(axis=(1, 2))



rmse = rmse.reshape(-1, 200) / (mask_range * (128*128)).repeat(200).reshape(-1, 200)
rmse = rmse.reshape(16, -1).sum(axis=1) / 200
rmse = rmse ** .5

ax1.plot(mask_range, rmse, marker="o", color="black")
ax1.set_xlabel("Percentage of Image Masked")
ax1.set_ylabel("Test RMSE Score")

plt.gcf().set_size_inches(6, 5)
plt.tight_layout()
plt.savefig("mask_hallucination_rmse.png")

plt.close("all")


ax2 = plt.subplot()

data = np.load("mr_data_simple.npy")
complex_data = np.load("mr_data.npy")

min_data = data.min()
min_complex = complex_data.min()

data -= min_data
complex_data -= min_complex

ax2.plot(mask_range, data * mask_range + min_data, marker="o", color="black")
ax2.plot(mask_range, complex_data * mask_range + min_complex, marker="o", color="black")

ax2.set_ylabel("Test MR Score")
ax2.set_xlabel("Percentage of Image Masked")

plt.gcf().set_size_inches(6, 5)
plt.tight_layout()


plt.savefig("mask_hallucination_mr.png")

