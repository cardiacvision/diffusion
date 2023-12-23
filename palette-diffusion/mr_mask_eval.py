import numpy as np
import matplotlib.pyplot as plt


# example of calculating the frechet inception distance in Keras
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize

gts = np.load("experiments_2d_temporal/test_inpainting_2d_simple_230808_220813/results/test/0/gts.npy") / 255
preds = np.load("experiments_2d_temporal/test_inpainting_2d_simple_230808_220813/results/test/0/results.npy") / 255

from skimage.transform import resize
from tqdm import tqdm
from MR_perceptual import test_network

def scale_images(images, new_shape):
    images_list = list()
    for image in tqdm(images):
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)

def calculate_mr(images1, images2):
    # calculate activations
    mr = []
    for i in tqdm(range(len(images1))):
        mr.append(test_network.MRPL_2_images(images1[i], images2[i]).detach().cpu().numpy()[0][0][0][0])
    return np.array(mr).mean()

mask_range = np.arange(0.05, 0.81, 0.05)

gts = gts.reshape(-1, 200, 128, 128)
preds = preds.reshape(-1, 200, 128, 128)

data = []

# for layer in range(16):
#     w = int((128**2 * mask_range[layer]) ** .5)
#     gt = np.repeat(gts[layer].reshape(200, 128, 128, 1)[:, (64 - w//2):(64 + w//2), (64 - w//2):(64 + w//2)], 3, axis=-1)
#     pred = np.repeat(preds[layer].reshape(200, 128, 128, 1)[:, (64 - w//2):(64 + w//2), (64 - w//2):(64 + w//2)], 3, axis=-1)
#     gt = scale_images(gt, (64, 64, 3))
#     pred = scale_images(pred, (64, 64, 3))
#     data.append(calculate_mr(gt, pred))

# data = np.array(data)
data = np.load("mr_data_simple.npy")
complex_data = np.load("mr_data.npy")

# data -= min(data.min(), complex_data.min())
# complex_data -= min(data.min(), complex_data.min())
# data /= mask_range
# complex_data /= mask_range

plt.plot(mask_range, data, marker="o")
plt.plot(mask_range, complex_data, marker="o")

plt.xlabel("Percentage of Image Masked")
plt.ylabel("Test MR Score")

plt.savefig("hallucination_simple_complex_mr.png")