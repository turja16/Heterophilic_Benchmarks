import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from GBKGNN.utils.metric import accuracy, roc_auc, compute_sigma_acc


# not for nclass=2
def train(args, dataset, device, model, optimizer, similarity, split_id):
    model.train()
    if len(dataset['graph'][0].train_mask.shape) != 1:
        train_mask = dataset['graph'][0].train_mask[:, split_id]
    else:
        train_mask = dataset['graph'][0].train_mask
    #
    optimizer.zero_grad()
    loss = 0
    # choose metric
    if dataset['num_classes'] > 2:
        metric = accuracy
    else:
        metric = roc_auc
    # for i in range(len(dataset['graph'])):
    data = dataset['graph'][0].to(device)
    out = model(data)
    regularizer_list = []
    if args.aug == True:
        out, sigma_list = out
        loss_function = CrossEntropyLoss()
        edge_train_mask = data.train_mask[data.edge_index[0]] * data.train_mask[data.edge_index[1]]
        if len(edge_train_mask.shape) == 2:
            edge_train_mask = torch.unbind(edge_train_mask, dim=1)[0]
        for sigma in sigma_list:
            sigma = sigma[edge_train_mask]
            # sigma_ = sigma.clone()
            # for i in range(len(sigma)):
            #     sigma_[i] = 1 - sigma[i]
            sigma_ = sigma.clone()
            sigma_ = 1 - sigma
            sigma = torch.cat(
                (sigma_.unsqueeze(1), sigma.unsqueeze(1)), 1)
            regularizer = loss_function(
                sigma.cuda(), torch.tensor(similarity, dtype=torch.long).cuda()[edge_train_mask])
            regularizer_list.append(regularizer)
            # training loss
        loss += F.nll_loss(
            out[train_mask],
            data.y[train_mask]) + args.lamda * sum(regularizer_list)
    else:
        # training loss
        loss += F.nll_loss(
            out[train_mask], data.y[train_mask])
    if len(dataset['graph']) == 1:  # single graph
        metric_train = metric(out[train_mask], data.y[train_mask])
    else:
        metric_train += metric(out[train_mask], data.y[train_mask]) * len(data.y)
    if not len(dataset['graph']) == 1:
        metric_train = metric_train / dataset['num_node']
    # update
    loss.backward()
    optimizer.step()
    del sigma, sigma_list
    # print('loss: {:.4f}'.format(loss.item()))
    return loss, metric_train


def test(args, dataset, device, model, similarity, split_id, mask_type="test"):
    model.eval()
    #
    # choose metric
    if dataset['num_classes'] > 2:
        metric = accuracy
    else:
        metric = roc_auc
    #
    if mask_type == "test":
        if len(dataset['graph'][0].test_mask.shape) != 1:  # multiple splits
            mask = dataset['graph'][0].test_mask[:, split_id]
        else:
            mask = dataset['graph'][0].test_mask
    if mask_type == "val":
        if len(dataset['graph'][0].val_mask.shape) != 1:  # multiple splits
            mask = dataset['graph'][0].val_mask[:, split_id]
        else:
            mask = dataset['graph'][0].val_mask
    #
    assert len(dataset['graph']) == 1
    data = dataset['graph'][0].to(device)
    if args.aug == True:
        # out, pred = model(data)[0].max(dim=1)
        # out, _ = model(data)[0].max(dim=1)
        out, _ = model(data)
        sigma0 = model(data)[1][0].tolist()
        sigma_acc = compute_sigma_acc(sigma0, similarity)
        # print('Sigma Accuracy: {:.4f}'.format(sigma_acc))
    else:
        out = model(data)
        # _, pred = model(data)
    #
    metric_test = metric(out[mask], data.y[mask])
    # print('{} Accuracy: {:.4f}'.format(mask_type, metric_test))
    return metric_test


#
def training(args, dataset, device, model, optimizer, similarity, split_id):
    best_val_acc = test_acc = 0
    counter: int = args.patience
    for epoch in range(args.epoch_num):
        loss, train_acc = train(args, dataset, device, model, optimizer, similarity, split_id)
        val_acc = test(args, dataset, device, model, similarity, split_id, mask_type="val")
        tmp_test_acc = test(args, dataset, device, model, similarity, split_id, mask_type="test")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            # reset counter
            counter = args.patience
        else:
            counter -= 1
        #
        if counter <= 0:
            print(f'Early stopping on epoch {epoch}.')
            break
        #
        if (epoch + 1) % args.log_interval == 0:
            print('epoch: {:03d}'.format(epoch + 1),
                  'loss: {:.4f}'.format(loss),
                  'train_acc: {:.4f}'.format(train_acc),
                  'val_acc: {:.4f}'.format(val_acc),
                  'test_acc: {:.4f}'.format(tmp_test_acc),
                  'final_test_acc: {:.4f}'.format(test_acc)
                  )
    print('*' * 10)
    print('Final_test_acc: {:.4f}'.format(test_acc))
    return test_acc
