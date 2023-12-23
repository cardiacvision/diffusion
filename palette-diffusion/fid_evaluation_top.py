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

diffusion_unet_output = np.load("experiments_3d_unet_top/finetune_spiral_3d_unetbaseline_230330_094145/results/test/0/results.npy")
diffusion_output = np.load("experiments_3d_40/test_spiral_3d_230129_195724/results/test/0/top_no_unet.npy")
unet_output = np.load("preprocess/top/test_data.npy")

from skimage.transform import resize
from tqdm import tqdm
gt_large = np.load("/mnt/data1/shared/data/option4-aniso/40/test_y.npy")

def scale_images(images, new_shape):
    images_list = list()
    for image in tqdm(images):
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)

def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_ssdiff(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)

    ssdiff = numpy.sum((act1 - act2)**2.0) / np.sum(act1.shape)
    return ssdiff

model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

indicies = np.arange(len(gt_large))
np.random.shuffle(indicies)

diffusion_unet = diffusion_unet_output.astype('float32')[indicies[:500]]*255
diffusion = diffusion_output.astype('float32')[indicies[:500]]*255
unet = unet_output.astype('float32')[indicies[:500]]*255
gt = gt_large.astype('float32')[indicies[:500]]*255

diffusion = scale_images(diffusion, (299,299,40))
diffusion_unet = scale_images(diffusion_unet, (299,299,40))
unet = scale_images(unet[:, :, :, 5:], (299,299,40))
gt = scale_images(gt, (299, 299, 40))

layer_diffusion = []
layer_diffusion_unet = []
layer_unet = []

for layer in range(40):
    du = preprocess_input(np.repeat(diffusion_unet[:, :, :, layer:layer+1], 3, axis=-1))
    d = preprocess_input(np.repeat(diffusion[:, :, :, layer:layer+1], 3, axis=-1))

    u = preprocess_input(np.repeat(unet[:, :, :, layer:layer+1], 3, axis=-1))
    g = preprocess_input(np.repeat(gt[:, :, :, layer:layer+1], 3, axis=-1))

    layer_diffusion.append(calculate_fid(model, d, g))
    layer_diffusion_unet.append(calculate_fid(model, du, g))
    layer_unet.append(calculate_fid(model, u, g))

plt.plot(range(1, 41), layer_diffusion, label="diffusion", marker="o")
plt.plot(range(1, 41), layer_diffusion_unet, label="diffusion with unet", marker="o")
plt.plot(range(1, 41), layer_unet, label="unet", marker="o")

plt.xlabel("Layer Number")
plt.ylabel("Test FID Score")

plt.legend()

plt.savefig("fid_diff_unet_baseline_top.png")