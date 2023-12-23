import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

CROP_PIXELS = 70
IMAGE_DIMS = (128, 128, 1)
VAL_SPLIT = 0.2
DATA_SIZE = 20000
BATCH_SIZE = 32

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, 128, 128, 1])

    return cropped_image[0], cropped_image[1]


@tf.function()
def random_jitter(input_image, real_image):
    # Resizing to 286x286
    input_image, real_image = resize(input_image, real_image, 144, 144)

    # Random cropping back to 256x256
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

raw_data_train_u = np.load("../data/uv_data/training_dataset_u_20000samples_5samples_temp5.npy")
raw_data_train_v = np.load("../data/uv_data/training_dataset_v_20000samples_5samples_temp5.npy")
raw_data_train = np.stack((raw_data_train_u, raw_data_train_v), axis=-1).reshape(raw_data_train_u.shape[:-1] + (-1,))
del raw_data_train_u, raw_data_train_v

raw_data_test = raw_data_train[18000:]
raw_data_train = raw_data_train[:18000]

raw_data_test_gt = raw_data_test[:, :, :, 0:2]
raw_data_train_gt = raw_data_train[:, :, :, 0:2]
with tf.device('/device:CPU:0'):
    train_dataset = tf.data.Dataset.from_tensor_slices((raw_data_train, raw_data_train_gt))
    test_dataset = tf.data.Dataset.from_tensor_slices((raw_data_test, raw_data_test_gt))

# del raw_data_train, raw_data_test, raw_data_train_u, raw_data_train_v
def mask_image(image, gt):
    # random_x = np.random.randint(0, IMAGE_DIMS[0] - CROP_PIXELS)
    # random_y = np.random.randint(0, IMAGE_DIMS[1] - CROP_PIXELS)
    h, w = 128, 128
    mask = np.ones(image.shape)
    mask[h//4: 3 * h//4, w//4: 3 * w//4] = 0

    mask = mask.astype(np.float32)

    return tf.multiply(image, mask), gt

def autoencoder(image):
    return image, image
    
train_dataset = train_dataset.map(mask_image, num_parallel_calls=tf.data.AUTOTUNE)
# train_dataset = train_dataset.map(random_jitter, num_parallel_calls=tf.data.AUTOTUNE)

test_dataset = test_dataset.map(mask_image, num_parallel_calls=tf.data.AUTOTUNE)

train_dataset = train_dataset.shuffle(200).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).cache()
test_dataset = test_dataset.shuffle(200).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).cache()
