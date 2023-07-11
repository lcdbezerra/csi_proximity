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

def thresh_to_class_str(lst):
    assert(sorted(lst) == lst)
    
    str_lst = ["d < "+str(lst[0])]
    for i in range(len(lst)-1):
        str_lst.append(str(lst[i])+" < d < "+str(lst[i+1]))
    str_lst.append(str(lst[-1])+" < d")
    return str_lst
    
def load_dataset(source):
    # LOAD DATASET
    ML_PATH = "../dataset/"
    if source=="1":
        filename = "Dataset_ISL_L30_S30_max30s_skipequal_20-03-2022.mat"
    elif source=="1+2":
        filename = "Dataset_ISL+AHM_L30_S30_max30s_skipequal_19-03-2022.mat"
    elif source=="2":
        filename = "Dataset_AHM_L30_S30_max30s_skipequal_28-03-2022.mat"
    else:
        raise ValueError("Dataset source not identified")
    f = h5py.File(ML_PATH+filename)
    s = f["super_s_filt"]

    var = {}
    for n,o in s.items():
        var[n] = np.array(o)

    f.close()
    return var

def dataset_digitize(data, seed, config):
    distmat = data["distmat"]
    X1 = data["X1"]
    X2 = data["X2"]
    p2p_flag = data["p2p_flag"]
    isl_flag = data["island_flag"]
    
    thresholds = config.dist_threshold
    train_test_split_strategy = config.train_test_split_strategy
    multiple_antennas = config.multiple_antennas
    bs = config.batch_size
    
    # Multi Antennas Support        
    if multiple_antennas:
        L = X1.shape[0]

        iset = range(2,6)
        jset = range(2,6)
        ijset = [(i,j) for i in iset for j in jset if i!=j]
        rep = len(ijset)
        # later: skip equal antennas only if P2S
        X1 = np.concatenate(tuple( [X1[:,i,:] for i,j in ijset] ), axis=0)
        X2 = np.concatenate(tuple( [X2[:,j,:] for i,j in ijset] ), axis=0)

        assert(rep*L == X1.shape[0])
        p2p_flag = np.tile(p2p_flag, (rep,1))
        distmat  = np.tile(distmat, (rep,1))
        # std_distmat = np.tile(std_distmat, (rep,1))
        isl_flag = np.tile(isl_flag, (rep,1))
        # dtimemat  = np.tile(dtimemat, (rep,1))
    else:
        X1 = X1[:,2,:]
        X2 = X2[:,2,:]
        # X = np.concatenate( (X1,X2), axis=0)
    
    valid_indices = np.logical_not(np.logical_or.reduce( (np.isnan(distmat), 
                                                          np.any(np.isnan(X1), axis=1).reshape((-1,1)), 
                                                          np.any(np.isnan(X2), axis=1).reshape((-1,1)),
                                                         ) )).reshape((-1,))
    distmat = distmat[valid_indices]
    X1 = X1[valid_indices,:]
    X2 = X2[valid_indices,:]
    p2p_flag = p2p_flag[valid_indices]
    isl_flag = isl_flag[valid_indices]
    
    # Sanity check
    assert(np.sum(np.isnan(distmat)) == 0)
    assert(np.sum(np.isnan(X1)) == 0)
    assert(np.sum(np.isnan(X2)) == 0)
    
    # Sepparate labels in below/above 
    Yb = multi_dist_threshold(distmat, thresholds)

    if train_test_split_strategy is None:
        X1t = torch.Tensor(X1)
        X2t = torch.Tensor(X2)
        Yb = torch.Tensor(Yb)
        # p2p_flag = torch.Tensor(p2p_flag, dtype=torch.bool) 

        ds = TensorDataset(X1t,X2t,Yb)
        # Split between training and test
        L = len(ds)
        trainL = int(np.floor(.8*L))
        train_ds, test_ds = random_split(ds, [trainL, L-trainL], generator=seed)
        # Create dataloaders
        train_dataloader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, drop_last=True)
        test_dataloader  = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4, drop_last=True)
        
    elif train_test_split_strategy==["P2P","P2P"]:
        p2p_flag = p2p_flag.astype(bool).reshape((-1,))
        # p2s_flag = np.logical_not(p2p_flag)
        
        X1t = torch.Tensor(X1[p2p_flag])
        X2t = torch.Tensor(X2[p2p_flag])
        Yb = torch.Tensor(Yb[p2p_flag])
        
        ds = TensorDataset(X1t,X2t,Yb)
        # Split between training and test
        L = len(ds)
        trainL = int(np.floor(.8*L))
        train_ds, test_ds = random_split(ds, [trainL, L-trainL], generator=seed)
        # Create dataloaders
        train_dataloader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, drop_last=True)
        test_dataloader  = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4, drop_last=True)
        
    elif train_test_split_strategy==["P2S","P2P"]:
        p2p_flag = p2p_flag.astype(bool).reshape((-1,))
        p2s_flag = np.logical_not(p2p_flag)

        test_ds = TensorDataset(torch.Tensor(X1[p2p_flag]),
                                torch.Tensor(X2[p2p_flag]),
                                torch.Tensor(Yb[p2p_flag]))
        
        train_ds = TensorDataset(torch.Tensor(X1[p2s_flag]),
                                 torch.Tensor(X2[p2s_flag]),
                                 torch.Tensor(Yb[p2s_flag]))
        
        # Create dataloaders
        train_dataloader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, drop_last=True)
        test_dataloader  = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4, drop_last=True)
        
    elif train_test_split_strategy==["AHM-P2S","P2P"]:
        p2p_flag = p2p_flag.astype(bool).reshape((-1,))
        p2s_flag = np.logical_not(p2p_flag)
        ahm_flag = np.logical_not( isl_flag.astype(bool).reshape((-1,)) )
        
        flag = p2s_flag & ahm_flag

        test_ds = TensorDataset(torch.Tensor(X1[p2p_flag]),
                                torch.Tensor(X2[p2p_flag]),
                                torch.Tensor(Yb[p2p_flag]))
        
        train_ds = TensorDataset(torch.Tensor(X1[flag]),
                                 torch.Tensor(X2[flag]),
                                 torch.Tensor(Yb[flag]))
        
        # Create dataloaders
        train_dataloader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, drop_last=True)
        test_dataloader  = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4, drop_last=True)
        
    elif train_test_split_strategy==["P2P+P2S","P2P"]:
        p2p_flag = p2p_flag.astype(bool).reshape((-1,))
        p2s_flag = np.logical_not(p2p_flag)
        
        X1p, X2p, Ybp = X1[p2p_flag], X2[p2p_flag], Yb[p2p_flag]
        inds = np.arange(X1p.shape[0])
        # np.random.seed(seed)
        np.random.shuffle(inds)
        
        Ltest = int(np.floor( (1-.8)*X1.shape[0] ))
        assert(Ltest<=inds.shape[0])
        
        test_ds = TensorDataset(torch.Tensor(X1p[inds[:Ltest]]),
                                torch.Tensor(X2p[inds[:Ltest]]),
                                torch.Tensor(Ybp[inds[:Ltest]])
                               )
        X1 = torch.Tensor(np.concatenate( (X1p[inds[Ltest:]], X1[p2s_flag]) , axis=0))
        X2 = torch.Tensor(np.concatenate( (X2p[inds[Ltest:]], X2[p2s_flag]) , axis=0))
        Yb = torch.Tensor(np.concatenate( (Ybp[inds[Ltest:]], Yb[p2s_flag]) , axis=0))
        train_ds = TensorDataset(X1,X2,Yb)
        
        # Create dataloaders
        train_dataloader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, drop_last=True)
        test_dataloader  = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4, drop_last=True)
        
    else:
        raise ValueError("Unknown strategy provided")

    return train_dataloader, test_dataloader


def binary_dist_threshold(distmat, threshold):
    Yb = np.zeros((distmat.shape[0], 2))
    Yb[:,0] = (distmat < threshold).reshape((-1,))
    Yb[:,1] = (distmat >=threshold).reshape((-1,))
    return Yb

def multi_dist_threshold(distmat, thresholds):
    # Check if thresholds are properly ordered
    assert(sorted(thresholds) == thresholds)
    
    # Digitize values, Yi stores index of class the values belong to
    Yi = np.digitize(distmat, thresholds)
    # Then convert into one-hot vectors
    Yb = np.eye(len(thresholds)+1)[Yi].squeeze()
    
    # SANITY CHECK
    _, Ytemp = torch.max(torch.tensor(Yb),1)
    try:
        assert(np.array_equal(Ytemp.numpy().reshape((-1,1)), Yi))
    except AssertionError:
        print("TEST FAILED")
        print(Ytemp.numpy().reshape((-1,1)).shape)
        print(Yi.shape)
        
        print(np.sum(Ytemp.numpy().reshape((-1,1)) != Yi))
        
        print(Ytemp.numpy().reshape((-1,1)))
        print(Yi)
        raise AssertionError("Dataset sanity check failed")
    print("DATASET TEST PASSED")
                          
    return Yb

def weighed_loss_fun(config, ds):
    method = config.loss_weighting_method
    if method == None:
        return nn.CrossEntropyLoss()
    elif method == "inv_class_frequency":
        sum_classes = np.sum(multi_dist_threshold(ds["distmat"], config.dist_threshold), 0)
        N = np.sum(sum_classes)
        weights = N/sum_classes # 1/P[class]
        weights = weights/len(config.dist_threshold)
        return nn.CrossEntropyLoss(weight=torch.Tensor(weights).to(device), reduction="mean")
        
    elif method == "inv_eff_no_samples":
        sum_classes = np.sum(multi_dist_threshold(ds["distmat"], config.dist_threshold), 0)
        beta = 0.999
        weights = (1-beta)/(1-np.power(beta,sum_classes))
        weights = weights/len(config.dist_threshold)
        # return nn.CrossEntropyLoss(weight=torch.Tensor(weights).to(device), reduction="mean")
        return nn.CrossEntropyLoss(weight=torch.Tensor(weights).to(device), reduction="sum")
    else:
        raise ValueError("Unrecognized loss weighing method")

# IMPLEMENT NEURAL NETWORK
import torch.nn as nn
import torch.nn.functional as F

class NN_Module(nn.Module):
    def __init__(self,config, n_inputs, n_outputs):
        super().__init__()
        
        self.config = config
        self.torch_seed = torch.Generator().manual_seed(config.seed)
        self.bs = config.batch_size
        self.lr = config.learning_rate
        self.epochs = config.epochs
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        self.BN0 = nn.BatchNorm1d(self.n_inputs)
        self.FC1 = nn.Linear(self.n_inputs, config.fc1)
        if config.fc1_bn: self.BN1 = nn.BatchNorm1d(config.fc1)
        self.FC2 = nn.Linear(config.fc1, config.fc2)
        if config.fc2_bn: self.BN2 = nn.BatchNorm1d(config.fc2)
        self.FC3 = nn.Linear(config.fc2, config.fc3)
        if config.fc3_bn: self.BN3 = nn.BatchNorm1d(config.fc3)
        self.FC4 = nn.Linear(config.fc3, self.n_outputs)
        
    def forward(self, x):
        x = self.BN0(x)
        x = F.leaky_relu(self.FC1(x))
        if self.config.fc1_bn: x = self.BN1(x)
        x = F.leaky_relu(self.FC2(x))
        if self.config.fc2_bn: x = self.BN2(x)
        x = F.leaky_relu(self.FC3(x))
        if self.config.fc3_bn: x = self.BN3(x)
        x = F.softmax(self.FC4(x))
        return x
    
def train(config=None):
    mode = "online" if online else "offline"
    with wandb.init(config=config, mode=mode) as run:
        
        config = wandb.config
        np.random.seed(config.seed)
        torch_seed = torch.Generator().manual_seed(config.seed)
        class_names = thresh_to_class_str(config.dist_threshold)
        train_dataloader, test_dataloader = dataset_digitize(ds[config.dataset],
                                                             torch_seed,
                                                             config)
        # print("AFTER DATASET")
        # breakpoint()
        sample = next(iter(train_dataloader))
        L = sample[0].shape[1]
        Lout = sample[2].shape[1]
        # print(L)
        # print(f"Lout:{Lout}")
        NN = NN_Module(config, 2*L, Lout).to(device)
        wandb.watch(NN, log="all")
        
        # Sanity check
        x = torch.cat((sample[0], sample[1]), dim=1).to(device)
        try:
            y = NN(x)
        except:
            raise ValueError("NN not compatible with provided data")
        
        
        # loss_fn = nn.CrossEntropyLoss()
        loss_fn = weighed_loss_fun(config,ds)
        
        if config.optimizer == "sgd":
            optimizer = optim.SGD(NN.parameters(), 
                                  lr=config.learning_rate)
        elif config.optimizer == "adam":
            optimizer = optim.Adam(NN.parameters(),
                                   lr=config.learning_rate)
        else:
            raise ValueError("Unrecognized optimizer")
        
        # for e in range(2):
        for e in range(config.epochs):
            NN.train()
            running_loss = 0
            
            for i, data in enumerate(tqdm(train_dataloader,
                                         desc=f"Epoch {e}/{config.epochs}"), 
                                     start=0):
                
                X = torch.cat((data[0],data[1]), dim=1).to(device)
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
            if e % 10 == 0:
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
                        X = torch.cat((data[0],data[1]), dim=1).to(device)
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
        
        save_path = scratch_dir + run.id + "/"
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
scratch_dir = f"../scratch/"

parser = argparse.ArgumentParser(description="Gridworld Training Script")

if __name__ == "__main__":
    ds = {i: load_dataset(i) for i in ["1", "1+2", "2"]}

    parser.add_argument("wandb_sweep", type=str, help="WANDB Sweep ID")
    parser.add_argument("-o", "--online", action="store_true", help="Upload experiment to WANDB")
    parser.add_argument("-c", "--count", type=int, default=0, help="Run count")
    args = parser.parse_args()

    run_count = args.count if args.count > 0 else None
    online = args.online
    wandb.agent(args.wandb_sweep, train, count=run_count)