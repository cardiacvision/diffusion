import numpy as np
import PIL.Image as Image
from tqdm import tqdm
# raw_data_train = np.load("../data/trainingdataset_03_u_20000samples_Ncp10_temp5_shuffled.npy")

# raw_data_train = np.transpose(raw_data_train, (0, 3, 1, 2))
# raw_data_train = raw_data_train[:, 0]
# raw_data_train = raw_data_train.reshape(-1, 128, 128)


raw_data_test = np.load("../data/validationdataset_03_u_5000samples_Ncp10_temp5.npy")
raw_data_test = np.transpose(raw_data_test, (0, 3, 1, 2))
raw_data_test = raw_data_test[:, 0]
raw_data_test = raw_data_test[np.random.choice(len(raw_data_test), 100)]
# raw_data_test = raw_data_test.reshape(-1, 128, 128)[:60]

# for idx, img in tqdm(enumerate(raw_data_train)):
#     image = Image.fromarray(np.uint8(img*255))
#     image.save(f"./train/{idx}.jpg")


for idx, img in tqdm(enumerate(raw_data_test)):
    image = Image.fromarray(np.uint8(img*255))
    image.save(f"./test/{idx}.jpg")

# for i in range(10):
#     k = i*10
#     for j in range(8):
#         image = Image.fromarray(np.uint8(raw_data_test[i] * 255))
#         image.save(f"./test_repeats/{k+j}.jpg")