# (📑 UNDER REVIEW) CSI-based Proximity Estimation: Data-driven and Model-based Approaches

Official code and dataset. Implementations are partly made on MATLAB and partly on Python with PyTorch.

## Instructions

### Python Environment Setup

### Data-Driven

The hyperparameter sweep across variants of the proposed neural network architecture is based on [Weights & Biases](https://wandb.ai). To create the sweep, run:

```
cd data_driven
python NN_sweep.py
```

which also prints the sweep ID. Then run an agent (or multiple) to start training indefinitely. Add `-c 1` to the end of the command below to run a single hyperparameter combination.

```
python NN_agent.py SWEEP_ID --online
```

### Model-Based

