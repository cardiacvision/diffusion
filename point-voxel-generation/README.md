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
```

or, install the pip requirements in `requirements.txt`.

Install PyTorchEMD by
```
cd metrics/PyTorchEMD
python setup.py install
cp build/**/emd_cuda.cpython-36m-x86_64-linux-gnu.so .
```

## Data

We uploaded the dataset to (Zenodo)[zenodo.org]

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