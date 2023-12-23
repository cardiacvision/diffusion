from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from data_next_timestep import NextTimeStep2DNPY
from tqdm import tqdm
import tensorflow as tf
import matplotlib as mpl

mode = "simple"

model = tf.keras.models.load_model("./unet_next_timestep.h5")

avg_rmse = np.zeros(2495)
for file in glob(f"../data/next-timestep-eval/{mode}*"):
    phase_loader = NextTimeStep2DNPY(data_path=f"{file}/u_data.npy", image_size=64)
    # iterator = iter(phase_loader)
    # curr_dict = next(iterator)
    # starter = curr_dict["cond_image"]
    # import ipdb; ipdb.set_trace()
    all_data = []
    gt_data = []
    prev_output = None
    for i in tqdm(range(len(phase_loader))):
        cond_image, gt_image = phase_loader[i]
        if i % 5 != 0:
            continue
        if prev_output is not None:
            cond_image = prev_output
        output = model.predict(cond_image.reshape(1, 64, 64, 5), verbose=0)

        all_data.append(output[0])
        gt_data.append(gt_image)
        prev_output = output
    
    outputs = np.concatenate(all_data, axis=-1).transpose(2, 0, 1)
    gts = np.concatenate(gt_data, axis=-1).transpose(2, 0, 1)
    # import ipdb; ipdb.set_trace()
    skip = 30

    fig, ax = plt.subplots(3, len(outputs)//skip)
    for i in range(0, len(outputs) - skip, skip):
        ax[0][i//skip].imshow(gts[i], cmap=mpl.colormaps["magma"])
        ax[1][i//skip].imshow(outputs[i], cmap=mpl.colormaps["magma"])
        ax[2][i//skip].imshow(np.abs(gts[i] - outputs[i]), cmap=mpl.colormaps["magma"])
        
        ax[0][i//skip].set_axis_off()
        ax[1][i//skip].set_axis_off()
        ax[2][i//skip].set_axis_off()

    fig.set_size_inches(len(outputs)//skip, 2)
    # fig.tight_layout()
    fig.savefig(f"u-autoreg/u_test_autoreg_{mode[0]}{file[-1]}.png")
    np.save(f"u-autoreg/u_autoreg_data_{mode[0]}{file[-1]}_output.npy", outputs)
    np.save(f"u-autoreg/u_autoreg_data_{mode[0]}{file[-1]}_gts.npy", gts)

    rmse = ((gts - outputs) ** 2).sum(axis=(1, 2)) ** 0.5
    avg_rmse += rmse
# import ipdb;ipdb.set_trace()

fig, ax = plt.subplots()
ax.plot(range(len(rmse)), rmse/5/(64**2))
ax.set_xlabel("Timestep")
ax.set_ylabel("Test RMSE Score")
ax.set_title(f"{mode} Data")
# fig.set_size_inches(100//skip, 2)
# fig.tight_layout()
fig.savefig(f"u-autoreg/u_{mode}_next_timestep_rmse.png")
np.save(f"u-autoreg/u_{mode}_rmse.npy", rmse/5/(64**2))
# fig.savefig("t_autoregressive_output.png")