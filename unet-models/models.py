from keras_unet_collection import models
import tensorflow.keras as keras
import tensorflow as tf
from data_next_timestep import train_dataset, test_dataset
with tf.device('/device:GPU:0'):
    model = models.unet_plus_2d((None, None, 5), [32, 64, 128, 256], n_labels=5,
                                stack_num_down=1, stack_num_up=1,
                                activation='LeakyReLU', output_activation="Sigmoid", 
                                batch_norm=True, pool=False, unpool=False, deep_supervision=False, name='xnet')

    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.SGD(lr=1e-2))

    model.fit(train_dataset, validation_data=test_dataset, epochs=256)
    model.evaluate(test_dataset)

    model.save("unet_next_timestep.h5")

    output = model.predict(test_dataset)
    import numpy as np

    np.save("unet_next_timestep.npy", output)