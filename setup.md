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

