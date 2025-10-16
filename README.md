# CLAX Baseline Experiments
This repository contains the expectation maximization baseline experiments with [PyClick](https://github.com/markovi/PyClick) for our paper `CLAX: Fast and Flexible Neural Click Models in JAX` which can be found [here](https://github.com/philipphager/clax/tree/main).

## Setup
To setup PyClick, we use [Mamba](https://mamba.readthedocs.io/en/latest/) to create an environment with the [PyPy](https://pypy.org/) interpreter.

After installing Mamba, you can set up the dependencies for this work using:
```bash
mamba env create -f environment.yaml
```

On OSX with an Apple Silicon chip, you might need to run:
```bash
CONDA_SUBDIR=osx-64 mamba env create -f environment.yaml
```

Afterwards, enter the virtual environmnent using:
```bash
mamba activate clax-baselines
```

Next, download the [Yandex WSCD-2012](https://dl.acm.org/doi/10.1145/2124295.2124396) dataset and update the path to it's main file under: `config/config.yaml`


And lastly, execute the script:
```
chmod +x ./scripts/1-yandex-10m.sh && ./scripts/1-yandex-10m.sh
```

Optionally, if you want to execute the script on a SLURM cluster, you can use:

```
./scripts/1-yandex-10m.sh +launcher=slurm
```

You can adjust the SLURM config for your cluster under: `config/launcher/slurm.yaml`.
