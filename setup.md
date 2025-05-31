# Base Architecture Extraction from Carlini

## Overview of structure

To evaluate this on a cluster, be sure to clone this repository

```git clone https://github.com/dmcgrath19/SimpleExtraction.git```

```cd SimpleExtraction```

then install dependencies

```pip install -r requirements.txt```

you may need to create a job script, but here is an example of how to make it run for the perplexity/extraction

(install mamba separately bc order matters)
```pip install causal-conv1d```
```pip install mamba-ssm --no-build-isolation```
```pip install flash-attn ninja```


You are now ready to go! Here is a sample script

```python main.py --N 1000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-160m --corpus-path monology/pile-uncopyrighted```

For a breakdown of the possible args look below

## Command-Line Arguments

### Required Arguments

| Argument        | Type   | Description |
|----------------|--------|-------------|
| `--model1`      | `str`  | Hugging Face model name for the first model. |
| `--model2`      | `str`  | Hugging Face model name for the second model. |
| `--corpus-path` | `str`  | Path to the corpus dataset. |

### Optional Arguments

| Argument         | Type    | Default  | Description |
|------------------|---------|----------|-------------|
| `--N`            | `int`   | `1000`   | Number of samples to generate. |
| `--batch-size`   | `int`   | `10`     | Batch size for generation. |
| `--corpus-subset`| `str`   | `None`   | Data subset if using splitted data. |
| `--name-tag`     | `str`   | `None`   | Name tag used for output file naming. |
| `--random-seed`  | `int`   | `None`   | Random seed for dataset shuffling. |
| `--split`        | `str`   | `None`   | Split for dataset (e.g., `train`, `test`). |
| `--is-splitted`  | `flag`  | `False`  | Use if dataset is pre-split. |
| `--is-wmt`       | `flag`  | `False`  | Use if dataset follows WMT format. |
| `--is-local`     | `flag`  | `False`  | Use if loading from a local text file. |
| `--input-len`    | `int`   | `150`    | Default length for input prompts. |

### Example Usage

```bash
python main.py \
  --N 10000 \
  --batch-size 10 \
  --model1 state-spaces/mamba-130m-hf \
  --model2 state-spaces/mamba-130m-hf \
  --corpus-path monology/pile-uncopyrighted \
  --name-tag mamba-130-len450 \
  --input-len 450
