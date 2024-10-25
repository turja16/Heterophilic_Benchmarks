# import time
# https://github.com/Godofnothing/HeterophilySpecificModels/blob/main/FAGCN/src/train.py
import torch
import torch.nn.functional as F

@torch.no_grad()
def accuracy(pr_logits, gt_labels):
    return (pr_logits.argmax(dim=-1) == gt_labels).float().mean().item()

@torch.no_grad()
def roc_auc(pr_logits, gt_labels):
    return roc_auc_score(gt_labels.cpu().numpy(), pr_logits[:, 1].cpu().numpy()) 

def train(args, features, labels, nclass, train, val, test, net, optimizer):
    # main loop
    final_test = None
    counter = 0
    max_metric = 0.0
    metric = accuracy if nclass > 2 else roc_auc
    for epoch in range(args.epoch_num):
        net.train()
        logp = net(features)
        # train
        cla_loss = F.nll_loss(logp[train], labels[train])
        loss = cla_loss
        train_metric = metric(logp[train], labels[train])
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del logp
        # test
        net.eval()
        logp = net(features)
        test_metric = metric(logp[test], labels[test])
        val_metric = metric(logp[val], labels[val])
        #
        if max_metric < val_metric:
            max_metric = val_metric
            final_test = test_metric
            counter = 0
        else:
            counter += 1
        #
        if counter >= args.patience:
            print('early stop')
            break
    return final_test
