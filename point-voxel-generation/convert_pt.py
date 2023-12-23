import argparse
import torch
import numpy as np
from pathlib import Path

def main(input_path, output_path):
    samples = torch.load(input_path)
    out = Path(output_path)

    for i in range(len(samples)):
        heart = samples[i].cpu().numpy()
        np.savez_compressed(out / f"generated_oct30_16k_{i}.npz", XYZ=heart[:, :3], Voltage=heart[:, 3:4])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str, default="./readable")
    
    opt = parser.parse_args()

    main(opt.input_path, opt.output_path)