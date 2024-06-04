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
Each folder corresponds to the code for different portions of the paper, and has a corresponding requirements.txt file.
<!--
## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/YourGithubName/deep-learning-project-template

# install project   
cd deep-learning-project-template 
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```
-->
### Citation   
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
