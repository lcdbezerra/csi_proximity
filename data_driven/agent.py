import wandb
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import os
import argparse

from NN_agent import thresh_to_class_str, dataset_digitize, weighed_loss_fun, NN_Module, load_dataset

IBEX = True if os.path.exists("/ibex/") else False
usr_name = "camaral" if IBEX else os.getlogin()
online = None
wandb_root = "lucascamara/csi_proximity/"
base_dir = f"/home/{usr_name}/code/csi_proximity_private"
scratch_dir = f"/home/{usr_name}/scratch/runs/" if not IBEX else \
                f"/ibex/user/{usr_name}/runs/"
# sys.path.append(base_dir)

parser = argparse.ArgumentParser(description="Weights & Biases Tutorial")

DEFAULT_CONFIG = {
    "dist_threshold": [5],
    "loss_weighting_method": None,
    "test_set": "random",
    # "train_test_split_strategy": ["P2P+P2S", "P2P"],
    "train_test_split_strategy": ["AHM-P2S", "P2P"],
    "dataset": "1+2",
    "multiple_antennas": True,
    "optimizer": "adam",
    "fc1": 16,
    "fc1_bn": True,
    "fc2": 16,
    "fc2_bn": True,
    "fc3": 8,
    "fc3_bn": True,

    "encoder_type": "lstm_final",
    "hidden_size": 16,

    "epochs": 5,
    "seed": 72,
    "batch_size": 64,
    "learning_rate": 1e-3,

}

def agent(config=None, default=False):
    mode = "online" if online else "offline"
    with wandb.init(config=config, mode=mode) as run:
        config = DEFAULT_CONFIG if default else wandb.config
        try:
            np.random.seed(config["seed"])
        except:
            pass
        if IBEX: 
            try: 
                run.summary["ibex_job_id"] = os.environ["SLURM_JOBID"]
            except: pass

        ############################

        train_nn(config)

        ############################

def train_nn(config):
    torch_seed = torch.Generator().manual_seed(config["seed"])
    class_names = thresh_to_class_str(config["dist_threshold"])
    train_dataloader, test_dataloader = dataset_digitize(ds[config["dataset"]],
                                                            torch_seed,
                                                            config)
    # print("AFTER DATASET")
    # breakpoint()
    sample = next(iter(train_dataloader))
    L = sample[0].shape[1]
    Lout = sample[2].shape[1]
    # print(L)
    # print(f"Lout:{Lout}")
    NN = NN_Module(config, L, Lout).to(device)
    wandb.watch(NN, log="all")
    
    # Sanity check
    # x = torch.cat((sample[0], sample[1]), dim=1).to(device)
    x = torch.stack((sample[0], sample[1]), dim=-1).to(device)
    try:
        y = NN(x)
    except:
        raise ValueError("NN not compatible with provided data")
    
    
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = weighed_loss_fun(config,ds)
    
    if config["optimizer"] == "sgd":
        optimizer = optim.SGD(NN.parameters(), 
                                lr=config["learning_rate"])
    elif config["optimizer"] == "adam":
        optimizer = optim.Adam(NN.parameters(),
                                lr=config["learning_rate"])
    else:
        raise ValueError("Unrecognized optimizer")
    
    # for e in range(2):
    for e in range(config["epochs"]):
        NN.train()
        running_loss = 0
        
        for i, data in enumerate(tqdm(train_dataloader,
                                        desc=f"Epoch {e}/{config['epochs']}"), 
                                    start=0):
            
            # X = torch.cat((data[0],data[1]), dim=1).to(device)
            X = torch.stack((data[0],data[1]), dim=-1).to(device)
            Y = data[2].to(device)
            
            optimizer.zero_grad()
            
            Yh = NN(X)
            loss = loss_fn(Yh, Y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f'Training Loss: {running_loss}')
        wandb.log({"train_loss": running_loss}, step=e)
            
        # if (e!= 0) and (e % 10 == 0):
        if e % 10 == 0 or e==(config["epochs"]-1):
            # Test set
            NN.eval()
            correct = 0
            total = 0
            running_loss = 0
            
            preds  = None
            labels = None
            # p2p = None
            
            with torch.no_grad():
                for i, data in enumerate(tqdm(test_dataloader,
                                        desc=f"Testing!"), 
                                    start=0):
                    # X = torch.cat((data[0],data[1]), dim=1).to(device)
                    X = torch.stack((data[0],data[1]), dim=-1).to(device)
                    Y = data[2].to(device)

                    Yh = NN(X)
                    _, Ypred = torch.max(Yh.data, 1)
                    _, Ytrue = torch.max(Y.data, 1)
                    total += Y.size(0)
                    correct += (Ypred==Ytrue).sum().item()
                    
                    loss = loss_fn(Yh, Y)
                    running_loss += loss.item()
                    
                    preds  = Ypred.cpu().numpy() if (preds is None)  else np.concatenate((preds,  Ypred.cpu().numpy()), axis=0)
                    labels = Ytrue.cpu().numpy() if (labels is None) else np.concatenate((labels, Ytrue.cpu().numpy()), axis=0)
                    # p2p = data[3].cpu().numpy() if (p2p is None) else np.concatenate((p2p, data[3].cpu().numpy()), axis=0)
            
            acc = 100 * correct / total
            print(f'Test Accuracy: {acc:.2f}%')
            print(f'Test Loss: {running_loss}')
            wandb.log({"test_loss": running_loss,
                        "test_acc": acc}, step=e)
    
    save_path = scratch_dir + wandb.run.id + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(NN.state_dict(), save_path+"model.h5")
    wandb.save(save_path+"model.h5")
    
    # Confusion Matrix!
    wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                        y_true=labels,
                                                        preds=preds,
                                                        class_names=class_names)
                })

ds = {}
online = None
parser = argparse.ArgumentParser(description="CSI Proximity")

if __name__ == "__main__":
    ds = {i: load_dataset(i,path=base_dir+"/dataset/") for i in ["1", "1+2", "2"]}
    parser.add_argument("wandb_sweep", type=str, help="WANDB Sweep ID")
    parser.add_argument("-o", "--online", action="store_true", help="Upload experiment to WANDB")
    parser.add_argument("-c", "--count", type=int, default=0, help="Run count")
    try:
        args = parser.parse_args()
        default_config = False
    except:
        args = parser.parse_args(["csi_proximity/bw9en1u5"])
        default_config = True
    run_count = args.count if args.count > 0 else None

    sweep_id = args.wandb_sweep
    online = args.online
    if default_config:
        agent(default=True)
    else:
        wandb.agent(sweep_id, lambda *args, **kw: agent(default=default_config,*args, **kw), count=run_count)