# (📑 UNDER REVIEW) CSI-based Proximity Estimation: Data-driven and Model-based Approaches

Official code and dataset. Implementations are partly made on MATLAB and partly on Python with PyTorch.

## Instructions

### Python Environment Setup

We recommend using [Conda](https://docs.conda.io/en/latest/) for creating an isolated environment dedicated for running our code and ensuring that your own packages won't be disturbed. Installation guidelines can be found [here](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html). 

After installing Conda, create the environment from our required package list using:

```
conda env create -f env.yml
```

Then activate the environment using `conda activate csi_proximity` and you're ready to start.

### Data-Driven

The hyperparameter sweep across variants of the proposed neural network architecture is based on [Weights & Biases](https://wandb.ai). To create the sweep, run:

```
cd data_driven
python NN_sweep.py
```

which also prints the sweep ID and a link for accessing the sweep results later on. Then run an agent (or multiple) to start training indefinitely. Add `-c 1` to the end of the command below to run a single hyperparameter combination.

```
python NN_agent.py SWEEP_ID --online
```

### Model-Based

🚧 Under construction 🚧