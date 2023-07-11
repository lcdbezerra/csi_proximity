import wandb
import pprint

sweep_config = {
    "name": "Multiclass NN Classifier, L30 S30 T30, Multi-strategy, Skip Equal, Multi Antennas",
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "test_acc",
    },
    "parameters": {
        "dist_threshold": {
            "values": [
                [5,42,90],
                [5,50],
                [5],
            ],
        },
        "loss_weighting_method": {
            "values": [
                None,
                # "inv_class_frequency",
                # "inv_eff_no_samples",
            ],
        },
        "test_set": {
            "values": [
                "random",
                # "fixed",
            ],
        },
        "train_test_split_strategy": {
            "values": [
                # None, # regular, P2P+P2S - P2P+P2S
                ["P2P+P2S", "P2P"],
                ["P2P", "P2P"],
                ["P2S","P2P"],
                # ["AHM-P2S","P2P"],
            ],
        },
        "dataset": {
            "values": [
                "1",
                "1+2",
                # "2",
            ],
        },
        "seed": {
            "values": [10, 20, 30, 40, 50, 60],
        },
        "loss_weighting_method": {
            "value": None,
        },
        "multiple_antennas": {
            "value": True,
        },
        "optimizer": {
            "value": "adam",
        },
        "epochs": {
            "value": 100,
        },
        "seed": {
            "values": [10, 20, 30],
        },
        "batch_size": {
            "value": 64,
        },
        "learning_rate": {
            "value": 1e-3,
        },
        "epochs": {
            "value": 100,
        },
        "fc1": {
            "values": [8, 16, 32, 64],
        },
        "fc1_bn": {
            "values": [True, False],
        },
        "fc2": {
            "values": [16, 32, 64],
        },
        "fc2_bn": {
            "values": [True, False],
        },
        "fc3": {
            "values": [4, 8, 16],
        },
        "fc3_bn": {
            "values": [True, False],
        },
    },
}

if __name__=="__main__":
    pprint.pprint(sweep_config)
    sweep_id = wandb.sweep(sweep_config, project="csi_proximity")
    print(sweep_id)