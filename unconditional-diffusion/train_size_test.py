from inference import main
from glob import glob
from random import sample
import shutil
import os
from tqdm import tqdm
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
script_format = """accelerate launch --gpu_ids 0 train_unconditional.py \
  --train_data_dir {} \
  --resolution=128 \
  --output_dir="{}" \
  --train_batch_size=16 \
  --num_epochs={} \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --checkpoints_total_limit=10 \
  --save_model_epochs={} \
  --mixed_precision=bf16 \
  --save_images_epochs=5000000 \
  --checkpointing_steps={}
"""
total_iterations = 50000*60
save_model = 50000 * 5
checkpointing_steps = 50000*10000
training_sizes = [5000, 100, 500, 1000, 10000, 50000]
# training_sizes = [100, 1000, 500]
# training_sizes = [5000, 10000, 50000]

main_path = "/mnt/data_jenner/tanish/data/uv_exp_data/"
# main_path = "/mnt/data_jenner/tanish/data/uv_param_range_data"
files = list(glob(f"{main_path}/*"))
for size in training_sizes:
    print(f"Training size: {size}")
    # files_sample = sample(files, size)
    # dir = f"/mnt/data_jenner/tanish/data/uv_exp_range_data/samples{size}"
    outdir = f"ddpm-128-param-samples{size}"
    # if not os.path.exists(dir):
    #     os.makedirs(dir)
    # for file in tqdm(files_sample):
    #     shutil.copy2(file, dir)
    # with open("run.sh", "w") as file:
    #     file.write(script_format.format(dir, outdir, total_iterations // size, save_model // size, checkpointing_steps // size))
    # os.system("bash run.sh")
    gen_dir = f"./range_test_param/samples{size}"
    gen_dir_raw = f"./range_test_param/samples_raw_{size}"
    # gen_dir = f"./range_test_param/samples{size}"
    if not os.path.exists(gen_dir):
        os.makedirs(gen_dir)
    if not os.path.exists(gen_dir_raw):
        os.makedirs(gen_dir_raw)
    main(outdir, gen_dir, gen_dir_raw)
    # cmp_dir = "/mnt/data_jenner/tanish/data/uv_exp_range_data/samples50000"
    cmp_dir = "/mnt/data_jenner/tanish/data/uv_param_range_data/samples50000"
    
    # os.system(f"python -m pytorch_fid {cmp_dir} {gen_dir} --device cuda:0 > fid_scores/samples{size}.txt")
    os.system(f"python -m pytorch_fid {cmp_dir} {gen_dir_raw} --device cuda:0 > fid_scores_param/samples{size}.txt")