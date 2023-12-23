import matplotlib.pyplot as plt
import numpy as np

out = np.load("./unet_results.npy")
gt = np.load("./ground_truths.npy")

for i in range(len(gt)):
    prediction = out[i]
    tar = gt[i]
    diff = np.abs(prediction - tar)
    title = ['Ground Truth', 'Predicted Image', 'Difference']
    fig, ax = plt.subplots(16, 3, figsize=(5, 20))
    # rmse = np.sqrt(np.mean(np.square(prediction - tar), axis=(0, 1, 2)))
    for j in range(16):
        # import ipdb; ipdb.set_trace()

        # display_list = [tar[:, :], prediction[:, :], diff[:, :]]
        display_list = [tar[:, :, j:j+1], prediction[:, :, j:j+1], diff[:, :, j:j+1]]
        # display_list = [test_input[0][:, :, 0], tar[0][:, :, j], prediction[0][:, :, j], rmse]
        for k in range(3):
            ax[0][k].set_title(title[k])
            if k == 2:
                text = ax[j][k].text(140,60,f"depth {j}", size=12,
                        verticalalignment='center', rotation=270)

            # Getting the pixel values in the [0, 1] range to plot.
            ax[j][k].imshow(display_list[k], vmin=0, vmax=1, interpolation="none")
            ax[j][k].axis('off')
    fig.tight_layout()
    fig.savefig(f"results/All_{i}")
    plt.close(fig)