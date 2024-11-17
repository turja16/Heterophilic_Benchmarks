import argparse
from copy import deepcopy
from typing import NamedTuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import Adam

from JacobiConv.datasets import load_dataset
from JacobiConv.impl import PolyConv, models, GDataset, utils


@torch.no_grad()
def accuracy(pr_logits, gt_labels):
    return (pr_logits.argmax(dim=-1) == gt_labels).float().mean()


@torch.no_grad()
def roc_auc(pr_logits, gt_labels):
    return roc_auc_score(gt_labels.cpu().numpy(), pr_logits.cpu().numpy())


def split(split_id, splits_lst):
    global baseG, trn_dataset, val_dataset, tst_dataset
    trn_dataset = GDataset.GDataset(*baseG.get_split(splits_lst[split_id]["train"]))
    val_dataset = GDataset.GDataset(*baseG.get_split(splits_lst[split_id]["valid"]))
    tst_dataset = GDataset.GDataset(*baseG.get_split(splits_lst[split_id]["test"]))


def buildModel(output_channels,
               conv_layer: int = 10,
               aggr: str = "gcn",
               alpha: float = 0.2,
               dpb: float = 0.0,
               dpt: float = 0.0,
               **kwargs):
    if args.multilayer:
        emb = models.Seq([
            models.TensorMod(baseG.x),
            nn.Dropout(p=dpb),
            nn.Sequential(nn.Linear(baseG.x.shape[1], output_channels),
                          nn.ReLU(inplace=True),
                          nn.Linear(output_channels, output_channels)),
            nn.Dropout(dpt, inplace=True)
        ])
    elif args.resmultilayer:
        emb = models.Seq([
            models.TensorMod(baseG.x),
            nn.Dropout(p=dpb),
            nn.Linear(baseG.x.shape[1], output_channels),
            models.ResBlock(
                nn.Sequential(nn.ReLU(inplace=True),
                              nn.Linear(output_channels, output_channels))),
            nn.Dropout(dpt, inplace=True)
        ])
    else:
        emb = models.Seq([
            models.TensorMod(baseG.x),
            nn.Dropout(p=dpb),
            nn.Linear(baseG.x.shape[1], output_channels),
            nn.Dropout(dpt, inplace=True)
        ])
    from functools import partial
    frame_fn = PolyConv.PolyConvFrame
    conv_fn = partial(PolyConv.JacobiConv, **kwargs)
    if args.power:
        conv_fn = PolyConv.PowerConv
    if args.legendre:
        conv_fn = PolyConv.LegendreConv
    if args.cheby:
        conv_fn = PolyConv.ChebyshevConv
    if args.bern:
        conv = PolyConv.Bern_prop(conv_layer)
    else:
        if args.fixalpha:
            from bestHyperparams import fixalpha_alpha
            alpha = fixalpha_alpha[args.dataset]["power" if args.power else (
                "cheby" if args.cheby else "jacobi")]
        conv = frame_fn(conv_fn,
                        depth=conv_layer,
                        aggr=aggr,
                        alpha=alpha,
                        fixed=args.fixalpha)
    comb = models.Combination(output_channels, conv_layer + 1, sole=args.sole)
    gnn = models.Gmodel(emb, conv, comb).to(device)
    return gnn


def work(output_channels,
         loss_fn,
         score_fn,
         splits_lst,
         conv_layer: int = 10,
         aggr: str = "gcn",
         alpha: float = 0.2,
         lr1: float = 1e-3,
         lr2: float = 1e-3,
         lr3: float = 1e-3,
         wd1: float = 0,
         wd2: float = 0,
         wd3: float = 0,
         dpb=0.0,
         dpt=0.0,
         patience: int = 10000,
         split_id: int = 0,
         **kwargs):
    utils.set_seed(0)
    split(split_id, splits_lst)
    gnn = buildModel(output_channels, conv_layer, aggr, alpha, dpb, dpt, **kwargs)
    optimizer = Adam([{
        'params': gnn.emb.parameters(),
        'weight_decay': wd1,
        'lr': lr1
    }, {
        'params': gnn.conv.parameters(),
        'weight_decay': wd2,
        'lr': lr2
    }, {
        'params': gnn.comb.parameters(),
        'weight_decay': wd3,
        'lr': lr3
    }])
    val_score = 0
    early_stop = 0
    for epoch in range(1000):
        train_loss = utils.train(optimizer, gnn, trn_dataset, loss_fn)
        score, _ = utils.test(gnn, val_dataset, score_fn, loss_fn=loss_fn)
        if (epoch + 1) % 100 == 0:
            print("Train loss= {:.4f}".format(train_loss),
                  "Val metric= {:.4f}".format(score))
        if score >= val_score:
            early_stop = 0
            val_score = score
            best_params = deepcopy(gnn.state_dict())
        else:
            early_stop += 1
        if early_stop > patience:
            break
    gnn.load_state_dict(best_params)
    test_score, test_loss = utils.test(gnn, tst_dataset, score_fn, loss_fn=loss_fn)
    print("Test set results:",
          "loss= {:.4f}".format(test_loss),
          "metric= {:.4f}".format(test_score))
    return test_score


def train_alldata_jacobiconv(device: torch.device,
                             args: Union[NamedTuple, argparse.Namespace]):
    baseG, splits_lst = load_dataset(args.dataset)
    baseG.to(device)
    trn_dataset, val_dataset, tst_dataset = None, None, None
    output_channels = baseG.num_targets

    if output_channels == 1:
        loss_fn = F.binary_cross_entropy_with_logits
        score_fn = roc_auc
        baseG.y = baseG.y.float()
    else:
        loss_fn = F.cross_entropy
        score_fn = accuracy

    acc_list = []
    if args.dataset in ['genius', 'deezer-europe', 'penn94', 'arxiv-year', 'pokec', 'snap-patents', 'twitch-gamer']:
        num_splits = len(splits_lst)
    else:
        num_splits = args.run

    for split_id in range(num_splits):
        print(f'Split [{split_id + 1}/{num_splits}]')
        test_metric = work(
            output_channels,
            loss_fn,
            score_fn,
            splits_lst,
            conv_layer=10,
            aggr='gcn',
            alpha=args.alpha,
            lr1=args.lr,
            lr2=args.lr,
            lr3=args.lr,
            wd1=args.wd,
            wd2=args.wd,
            wd3=args.wd,
            dpb=args.dpb,
            dpt=args.dpt,
            patience=10000,
            split_id=split_id,
            a=args.a,
            b=args.b
        )
        acc_list.append(test_metric)

    if args.dataset in ['deezer-europe', 'genius', 'penn94', 'arxiv-year', 'pokec', 'snap-patents', 'twitch-gamer']:
        sub = 'large'
    elif args.dataset in ['Cora', 'CiteSeer', 'PubMed', 'chameleon', 'cornell', 'film', 'squirrel', 'texas',
                          'wisconsin']:
        sub = 'geom'
    elif args.dataset in ['squirrel_filtered', 'chameleon_filtered', 'roman_empire', 'minesweeper', 'questions',
                          'amazon_ratings', 'tolokers']:
        sub = 'critical'
    elif args.dataset in ['wiki_cooc', 'blogcatalog', 'flickr']:
        sub = 'opengsl'
    elif args.dataset in ['Bgp', ]:
        sub = 'pathnet'
    else:
        raise ValueError('Invalid data name')

    test_mean = np.mean(acc_list)
    test_std = np.std(acc_list)
    filename = f'./{args.method.lower()}_{sub}.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(f"{args.method.lower()}, " +
                        f"{args.dataset}, " +
                        f"{test_mean:.4f}, " +
                        f"{test_std:.4f}, " +
                        f"{args}\n")


if __name__ == "__main__":
    args = utils.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_alldata_jacobiconv(device, args)
