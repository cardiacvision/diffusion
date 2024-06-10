# Task 6: Unconditional generation of two-dimensional spiral wave patterns

## Requirements:

Make sure the following environments are installed.

```python
conda env create -f environment.yml
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

## Data

We uploaded the dataset to [Zenodo](https://zenodo.org)

Change the path in `script.sh` to point to your data directory.

## Training:

```bash
$ bash script.sh
```

Please refer to the python file for optimal training parameters.

## Testing:

```bash
$ python inference.py --model_path {PATH_TO_MODEL} --out_path {OUTPUT_DIR} --out_path_raw {OUTPUT_RAW_DIR}
```

This script outputs the u and v parameters in a human readable formate to `{OUTPUT_DIR}`. The png files with u as the r channel, and v as the g channel, and zeros in the blue channel to `{OUTPUT_RAW_DIR}`.

## Acknowledgement

Thanks to [huggingface/diffusers](https://github.com/huggingface/diffusers).