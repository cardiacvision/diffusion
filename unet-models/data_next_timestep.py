import numpy as np

from skimage.transform import resize
import tensorflow as tf


class NextTimeStep2DNPY:
    def __init__(self, data_path, image_size=128, until=None, skip=None, num_prev=5):
        # self.x = np.load(data_x)[:, :, :, ::7]
        self.image_size = (image_size, image_size)
        self.num_prev = num_prev
        self.images = np.load(data_path)
        self.images = self.images.reshape(-1, self.images.shape[1], self.images.shape[2], 1)
        if until is not None:
            self.images = self.images[:until]
        if skip is not None:
            self.images = self.images[skip:]



    def __getitem__(self, index):
        ret = {}
        # y = resize((self.y[index]/1.212)*255, self.image_size)
        x = []
        for i in range(self.num_prev):
            x.append(resize(np.array(self.images[i+index])[:, :, 0:1], self.image_size))
        x = np.concatenate(x, axis=-1).astype(np.float32)

        y = []
        for i in range(5):
            y.append(resize(np.array(self.images[i+self.num_prev+index])[:, :, 0:1], self.image_size))
        y = np.concatenate(y, axis=-1).astype(np.float32)
        assert 64 in y.shape
        # y = np.array(pil_loader(self.images[index+self.num_prev]))[:, :, 0:1]

        img = y
        cond_image = x
        

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        return (cond_image, img)

    def __len__(self):
        return len(self.images) - self.num_prev - 4

    def get_generator(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

train = NextTimeStep2DNPY(data_path="../data/flat.npy", skip=5000, image_size=64)
test = NextTimeStep2DNPY(data_path="../data/flat.npy", until=5000, image_size=64)

train_dataset = tf.data.Dataset.from_generator(
     train.get_generator,
     output_signature=(
         tf.TensorSpec(shape=(64, 64, 5), dtype=tf.float32),
         tf.TensorSpec(shape=(64, 64, 5), dtype=tf.float32))).prefetch(tf.data.AUTOTUNE).cache().shuffle(1000).batch(32)

test_dataset = tf.data.Dataset.from_generator(
     test.get_generator,
     output_signature=(
         tf.TensorSpec(shape=(64, 64, 5), dtype=tf.float32),
         tf.TensorSpec(shape=(64, 64, 5), dtype=tf.float32))).batch(32).cache()