import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import PIL.Image as Image
from tqdm import tqdm

test_images = np.random.randint(10678, size=1000) + 56000
train_images = np.random.randint(50000, size=3000)

u_data = []
for i in train_images:
    image = Image.open(f"../../../data/uv_exp_data/{i}.png")
    image = np.array(image)
    u = image[:, :, 0]
    v = image[:, :, 1]

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(u, cmap=mpl.colormaps["magma"])
    # ax[0].set_title("U variable")
    # ax[0].axis("off")

    # ax[1].imshow(v, cmap=mpl.colormaps["magma"])
    # ax[1].set_title("V variable")
    # ax[1].axis("off")


    # fig.savefig(f"training_images/trained{i}.png")
    # plt.close("all")
    u_data.append(u)
np.save("u_training_data.npy", u_data)

for i in test_images:
    image = Image.open(f"../../../data/uv_exp_data/{i}.png")
    image = np.array(image)
    u = image[:, :, 0]
    v = image[:, :, 1]

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(u, cmap=mpl.colormaps["magma"])
    # ax[0].set_title("U variable")
    # ax[0].axis("off")

    # ax[1].imshow(v, cmap=mpl.colormaps["magma"])
    # ax[1].set_title("V variable")
    # ax[1].axis("off")


    # fig.savefig(f"training_images/trained{i}.png")
    # plt.close("all")
    u_data.append(u)



np.save("u_testing_data.npy", u_data)