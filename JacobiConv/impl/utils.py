import numpy as np
import random
import torch
import argparse


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='')
    # Data settings
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--method', type=str, default='Jacobi')
    parser.add_argument('--run', type=int, default=10)
    # parser.add_argument('--split', type=str, default="default", 
    #                     choices=["dense", "sparse", "default"])
    # parser.add_argument('--split_id', type=int, default=0)                    

    # Train settings
    # parser.add_argument('--repeat', type=int, default=1)
    # parser.add_argument('--test', action='store_true')

    # Optuna Settings
    parser.add_argument('--optruns', type=int, default=50)
    parser.add_argument('--path', type=str, default="")
    parser.add_argument('--name', type=str, default="opt")

    # Model Settings
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--savemodel', action='store_true')
    parser.add_argument('--power', action="store_true")
    parser.add_argument('--cheby', action="store_true")
    parser.add_argument('--legendre', action="store_true")
    parser.add_argument('--bern', action="store_true")
    parser.add_argument('--sole', action="store_true")
    parser.add_argument('--fixalpha', action="store_true")
    parser.add_argument('--multilayer', action="store_true")
    parser.add_argument('--resmultilayer', action="store_true")

    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--a', type=float, default=0.0)
    parser.add_argument('--b', type=float, default=0.0)
    parser.add_argument('--dpb', type=float, default=0.0)
    parser.add_argument('--dpt', type=float, default=0.0)

    args = parser.parse_args()
    return args

def train(optimizer, model, ds, loss_fn):
    optimizer.zero_grad()
    model.train()
    pred = model(ds.edge_index, ds.edge_attr, ds.mask).squeeze()
    loss = loss_fn(pred, ds.y)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, ds, metrics, loss_fn=None):
    model.eval()
    pred = model(ds.edge_index, ds.edge_attr, ds.mask).squeeze()
    y = ds.y
    loss = loss_fn(pred, y)
    return metrics(pred, y).item(), loss.item()
