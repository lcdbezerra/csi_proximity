import wandb
import pprint

sweep_config = {
    "name": "Multiclass LDA Classifier, L30 S30 T30, multi-strategy, Skip Equal, Multi Antennas",
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "test_acc",
    },
    "parameters": {
        "dist_threshold": {
            "values": [
                # [10, 30, 80],
                # [3,9,21,42,90],
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
        "multiple_antennas": {
            "values": [
                # False, 
                True,
            ],
        },
        "dataset": {
            "values": [
                "ISL",
                "ISL+AHM",
                # "AHM",
            ],
        },
        "seed": {
            "values": [10, 20, 30, 40, 50, 60],
        },
    },
}

if __name__=="__main__":
    pprint.pprint(sweep_config)
    sweep_id = wandb.sweep(sweep_config, project="gpt-0")