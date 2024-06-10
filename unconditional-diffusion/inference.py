from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import argparse
import PIL.Image as im

def main(model_path, out_path, out_path_raw):
    generator = DiffusionPipeline.from_pretrained(model_path)
    generator.to("cuda")
    generator.scheduler = DPMSolverMultistepScheduler.from_config(generator.scheduler.config)

    u_data = []
    v_data = []
    images = []
    batch_size = 16
    for i in range(200):
        #print(len(generator(num_inference_steps=20, generator=rand_gens).images), len(rand_gens))
        for image in generator(num_inference_steps=20).images:
            image = np.array(image)
            u = image[:, :, 0]
            v = image[:, :, 1]

            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(u, cmap=mpl.colormaps["magma"])
            ax[0].set_title("U variable")
            ax[0].axis("off")

            ax[1].imshow(v, cmap=mpl.colormaps["magma"])
            ax[1].set_title("V variable")
            ax[1].axis("off")


            fig.savefig(f"{out_path}/generated{i}.png")
            u_data.append(u)
            v_data.append(v)
            plt.close("all")
            images.append(image)
            im.fromarray(image).save(f"{out_path_raw}/generated_raw{i}.png")

    np.save(f"{out_path}/u_data.npy", u_data)
    np.save(f"{out_path}/v_data.npy", v_data)
    np.save(f"{out_path}/images.npy", images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unconditional Diffusion Inference")
    parser.add_argument("--model_path", type=str, help="Path to the pre-trained model")
    parser.add_argument("--out_path", type=str, help="Output directory for generated images")
    parser.add_argument("--out_path_raw", type=str, help="Output directory for raw generated images")
    args = parser.parse_args()

    main(args.model_path, args.out_path, args.out_path_raw)