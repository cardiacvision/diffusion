from inference import main
from glob import glob
from random import sample
import shutil
import os
from tqdm import tqdm

training_sizes = [20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000]
# training_sizes = [100, 1000, 500]
# training_sizes = [5000, 10000, 50000]

for size in training_sizes:
    print(f"Ckpt iterations: {size}")
    outdir = f"ddpm-128-exp-samples50000/checkpoint-{size}"
    gen_dir = f"./range_test_ckpt/checkpoint{size}"
    gen_dir_raw = f"./range_test_ckpt/checkpoint_raw{size}"
    if not os.path.exists(gen_dir):
        os.makedirs(gen_dir)
    if not os.path.exists(gen_dir_raw):
        os.makedirs(gen_dir_raw)
    main(outdir, gen_dir, gen_dir_raw)
