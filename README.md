<div align="center">    
 
# Dreaming of Electrical Waves: Generative Modeling of Cardiac Excitation Waves using Diffusion Models     

[![https://arxiv.org/abs/2312.14830](http://img.shields.io/badge/paper-arxiv.2312.14830-B31B1B.svg)](https://arxiv.org/abs/2312.14830)
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->

</div>
 
## Description   
This repository contains all the code files used to generate the results from the paper. 
Each folder corresponds to the code for different portions of the paper, and has a corresponding requirements.txt file. Each folder also has an associated README.md file containing installation information and information on how to run the models.


| Tasks  | Base Folder                | Command to Run Training                                    | Comments |
|--------|----------------------------|------------------------------------------------------------|----------|
| Task 1 | [`palette-diffusion/`](https://google.com)       | `python run.py -c config/conditional.json -p train`        |          |
| Task 2 | `point-voxel-diffusion/`   | `python train_generation.py`                               |          |
| Task 3 | `palette-diffusion/`       | `python run.py -c config/next_timestep.json -p train`      |          |
| Task 4 | `palette-diffusion/`       | `python run.py -c config/spiral_3d.json -p train`          |          |
| Task 5 | `palette-diffusion/`       | `python run.py -c config/inpainting_2d_time.json -p train` |          |
| Task 6 | `unconditional-diffusion/` | `bash script.sh`                                           |          |

## Citation   
```
@article{baranwal2023dreaming,
      title={Dreaming of Electrical Waves: Generative Modeling of Cardiac Excitation Waves using Diffusion Models}, 
      author={Tanish Baranwal and Jan Lebert and Jan Christoph},
      year={2023},
      eprint={2312.14830},
      archivePrefix={arXiv},
      primaryClass={physics.med-ph}
}
```   

## Acknowledgements

We are benefiting a lot from the following projects:
- [Janspiry/Palette-Image-to-Image-Diffusion-Models](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)
- [alexzhou907/PVD](https://github.com/alexzhou907/PVD)
- [huggingface/diffusers](https://github.com/huggingface/diffusers)