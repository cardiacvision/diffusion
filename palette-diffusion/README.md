# Tasks 1, 3, 4, 5: Palette Image-to-Image Diffusion Models

This is a modified implementation of **Palette: Image-to-Image Diffusion Models** in **Pytorch**, and it is mainly inherited from the [implementation by Janspiry](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models).

## Requirements
```python
pip install -r requirements.txt
```

## Data

We uploaded the data for the different tasks to [Zenodo](https://zenodo.org)
- [Task 1: Generation of parameter-specific two-dimensional spiral waves](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256)
- [Task 3: Prediction of the evolution of spiral wave dynamics over time](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256)
- [Task 4: Reconstruction of three-dimensional scroll waves from two-dimensional surface observations](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256)
- [Task 5: Inpainting of two-dimensional spiral wave dynamics, see section](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256)

After you prepared own data, you need to modify the corresponding configure file to point to your data. Take the following as an example:

```yaml
"which_dataset": {  // import designated dataset using arguments 
    "name": ["data.dataset", "InpaintDataset"], // import Dataset() class
    "args":{ // arguments to initialize dataset
    	"data_root": "your data path",
    	"data_len": -1,
    	"mask_mode": "hybrid"
    } 
},
```

More choices about **dataloader** and **validation split** also can be found in `datasets`  part of configure file.

## Training

Run the script:

| Tasks  | Base Folder                | Command to Run Training                                    | Comments |
|--------|----------------------------|------------------------------------------------------------|----------|
| Task 1 | `palette-diffusion/`       | `python run.py -c config/conditional.json -p train`        |          |
| Task 3 | `palette-diffusion/`       | `python run.py -c config/next_timestep.json -p train`      |          |
| Task 4 | `palette-diffusion/`       | `python run.py -c config/spiral_3d.json -p train`          |          |
| Task 5 | `palette-diffusion/`       | `python run.py -c config/inpainting_2d_time.json -p train` |          |

More choices about **backbone**, **loss** and **metric** can be found in `which_networks`  part of configure file.

## Testing

1. Modify the configure file to point to your data following the steps in **Data** part.
2. Set your model path:

	Set `resume_state` of configure file to the directory of previous checkpoint. Take the following as an example, this directory contains training states and saved model:

	```yaml
	"path": { //set every part file path
		"resume_state": "experiments/checkpoint/100" 
	},
	```

3. Run the script:

| Tasks  | Base Folder                | Command to Run Testing                                    | Comments |
|--------|----------------------------|------------------------------------------------------------|----------|
| Task 1 | `palette-diffusion/`       | `python run.py -c config/conditional.json -p test`        |          |
| Task 3 | `palette-diffusion/`       | `python run.py -c config/next_timestep.json -p test`      |          |
| Task 4 | `palette-diffusion/`       | `python run.py -c config/spiral_3d.json -p test`          |          |
| Task 5 | `palette-diffusion/`       | `python run.py -c config/inpainting_2d_time.json -p test` |          |


## Acknowledgements
Our work is based on the following theoretical works:
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
- [Palette: Image-to-Image Diffusion Models](https://arxiv.org/pdf/2111.05826.pdf)
- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)

and we are benefiting a lot from the following projects:
- [Janspiry/Palette-Image-to-Image-Diffusion-Models](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)