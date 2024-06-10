# Task 2: Generation of scroll waves in bi-ventricular heart shapes

Modified Implementation of Shape Generation and Completion Through Point-Voxel Diffusion, inherited from [alexzhou907/PVD](https://github.com/alexzhou907/PVD).

## Requirements:

Make sure the following environments are installed.

```
python==3.6
pytorch==1.4.0
torchvision==0.5.0
cudatoolkit==10.1
matplotlib==2.2.5
tqdm==4.32.1
open3d==0.9.0
trimesh=3.7.12
scipy==1.5.1
torch-scatter==2.0.4
torch-sparse==0.6.1
torch-cluster==1.5.4
torch-spline-conv==1.2.0
descartes==1.1.0
fire==0.3.1
jupyter==1.0.0
opencv_python==4.3.0
Shapely==1.7.0
Pillow==6.2.1
torch_geometric==1.6.0
```

or, install using conda:

```python
conda env create -f environment.yml
```

Install PyTorchEMD by
```
cd metrics/PyTorchEMD
python setup.py install
cp build/**/emd_cuda.cpython-36m-x86_64-linux-gnu.so .
```

## Data

We uploaded the dataset to [Zenodo](https://zenodo.org)

## Training:

```bash
$ python train_generation.py
```

Please refer to the python file for optimal training parameters.

## Testing:

```bash
$ python test_generation.py --model MODEL_PATH
```

## Acknowledgement

Thanks to [alexzhou907/PVD](https://github.com/alexzhou907/PVD) for the PVD implementation.