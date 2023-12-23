import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

IMAGE_DIMS = (64, 64, 1)
VAL_SPLIT = 0.2
DATA_SIZE = 20000
BATCH_SIZE = 16

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def process(input, input2):
    input_image = tf.image.resize(input, [64, 64],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_image2 = tf.image.resize(input2, [64, 64],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, input_image2

train_x = np.load("../data/new/train_x_top.npy")[:, :, :, :5]
train_y = np.load("../data/new/train_y.npy")

test_x = np.load("../data/new/test_x_top.npy")[:, :, :, :5]
test_y = np.load("../data/new/test_y.npy")

with tf.device('/device:CPU:0'):
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))

# del raw_data_train, raw_data_test, raw_data_train_u, raw_data_train_v
    
train_dataset = train_dataset.map(process, num_parallel_calls=tf.data.AUTOTUNE)
# train_dataset = train_dataset.map(random_jitter, num_parallel_calls=tf.data.AUTOTUNE)

test_dataset = test_dataset.map(process, num_parallel_calls=tf.data.AUTOTUNE)

train_dataset = train_dataset.shuffle(200).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.shuffle(200).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
