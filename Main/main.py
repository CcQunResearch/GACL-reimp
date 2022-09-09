# -*- coding: utf-8 -*-
# @Time    : 2022/9/1 14:44
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : main.py
# @Software: PyCharm
# @Note    :
import sys
import os.path as osp

dirname = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(dirname, '..'))

import numpy as np
import time
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from Main.dataset import WeiboDataset, WeiboDatasetTest
from Main.bert import load_bert
from Main.pargs import pargs
from Main.utils import create_log_dict, write_log, write_json
from Main.sort import sort_weibo_dataset, sort_weibo_2class_dataset
from Main.model import GCN_Net
from torch.optim import Adam, SGD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train(train_loader, model, optimizer, optimizer_hard, epsilon, lamda, lamda_ad, device):
    model.train()
    total_loss = 0

    for batch_data in train_loader:
        optimizer.zero_grad()
        optimizer_hard.zero_grad()

        batch_data.to(device)
        pred, cl_loss, y = model(batch_data)
        cls_loss = F.binary_cross_entropy(pred, y.view(-1).to(torch.float32))
        loss = cls_loss + lamda * cl_loss
        total_loss += loss.item() * batch_data.num_graphs

        # loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()

        pred, cl_loss, y = model(batch_data)
        cls_loss = F.binary_cross_entropy(pred, y.view(-1).to(torch.float32))
        loss_ad = epsilon / (cls_loss + lamda_ad * cl_loss)
        loss_ad.backward()
        optimizer_hard.step()

    return total_loss / len(train_loader.dataset)


def test(model, dataloader, device):
    model.eval()
    error = 0

    y_true = []
    y_pred = []
    for data in dataloader:
        data = data.to(device)
        pred, _, y = model(data)
        error += F.binary_cross_entropy(pred, y.view(-1).to(torch.float32)).item() * data.num_graphs
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        y_true += y.tolist()
        y_pred += pred.tolist()
    acc = accuracy_score(y_true, y_pred)
    prec = [precision_score(y_true, y_pred, pos_label=1, average='binary'),
            precision_score(y_true, y_pred, pos_label=0, average='binary')]
    rec = [recall_score(y_true, y_pred, pos_label=1, average='binary'),
           recall_score(y_true, y_pred, pos_label=0, average='binary')]
    f1 = [f1_score(y_true, y_pred, pos_label=1, average='binary'),
          f1_score(y_true, y_pred, pos_label=0, average='binary')]
    return error / len(dataloader.dataset), acc, prec, rec, f1


def test_and_log(model, val_loader, test_loader, device, epoch, lr, loss, train_acc, log_record):
    val_error, val_acc, val_prec, val_rec, val_f1 = test(model, val_loader, device)
    test_error, test_acc, test_prec, test_rec, test_f1 = test(model, test_loader, device)
    log_info = 'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation BCE: {:.7f}, Test BCE: {:.7f}, Train ACC: {:.3f}, Validation ACC: {:.3f}, Test ACC: {:.3f}, Test PREC(T/F): {:.3f}/{:.3f}, Test REC(T/F): {:.3f}/{:.3f}, Test F1(T/F): {:.3f}/{:.3f}' \
        .format(epoch, lr, loss, val_error, test_error, train_acc, val_acc, test_acc, test_prec[0], test_prec[1],
                test_rec[0],
                test_rec[1], test_f1[0], test_f1[1])

    log_record['val accs'].append(round(val_acc, 3))
    log_record['test accs'].append(round(test_acc, 3))
    log_record['test prec T'].append(round(test_prec[0], 3))
    log_record['test prec F'].append(round(test_prec[1], 3))
    log_record['test rec T'].append(round(test_rec[0], 3))
    log_record['test rec F'].append(round(test_rec[1], 3))
    log_record['test f1 T'].append(round(test_f1[0], 3))
    log_record['test f1 F'].append(round(test_f1[1], 3))
    return val_error, log_info, log_record


if __name__ == '__main__':
    args = pargs()

    dataset = args.dataset
    device = args.gpu if args.cuda else 'cpu'
    runs = args.runs
    k = args.k
    bert_path = args.bert_path

    batch_size = args.batch_size
    epochs = args.epochs
    epsilon = args.epsilon
    lamda = args.lamda
    lamda_ad = args.lamda_ad

    label_source_path = osp.join(dirname, '..', 'Data', dataset, 'source')
    label_dataset_path = osp.join(dirname, '..', 'Data', dataset, 'dataset')
    train_path = osp.join(label_dataset_path, 'train')
    val_path = osp.join(label_dataset_path, 'val')
    test_path = osp.join(label_dataset_path, 'test')

    log_name = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(time.time()))
    log_path = osp.join(dirname, '..', 'Log', f'{log_name}.log')
    log_json_path = osp.join(dirname, '..', 'Log', f'{log_name}.json')

    log = open(log_path, 'w')
    log_dict = create_log_dict(args)

    for run in range(runs):
        write_log(log, f'run:{run}')
        log_record = {'run': run, 'val accs': [], 'test accs': [], 'test prec T': [], 'test prec F': [],
                      'test rec T': [], 'test rec F': [], 'test f1 T': [], 'test f1 F': []}

        if dataset == 'Weibo':
            sort_weibo_dataset(label_source_path, label_dataset_path, k)
        elif dataset == 'Weibo-2class' or dataset == 'Weibo-2class-long':
            sort_weibo_2class_dataset(label_source_path, label_dataset_path, k)

        bert_tokenizer, bert_model, bert_config = load_bert(bert_path)
        bert_model.to(device)

        train_dataset = WeiboDataset(train_path, bert_tokenizer, bert_model, probabilities=args.probabilities,
                                     droprate=args.droprate, join_source=args.join_source,
                                     mask_source=args.mask_source, drop_mask_rate=args.drop_mask_rate)
        val_dataset = WeiboDatasetTest(val_path, bert_tokenizer, bert_model, join_source=args.join_source)
        test_dataset = WeiboDatasetTest(test_path, bert_tokenizer, bert_model, join_source=args.join_source)

        bert_model.to('cpu')

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        model = GCN_Net(bert_config.hidden_size, args.hid_feats, args.out_feats, args.t).to(device)
        for para in model.hard_fc1.parameters():
            para.requires_grad = False
        for para in model.hard_fc2.parameters():
            para.requires_grad = False
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=args.lr, weight_decay=args.weight_decay)

        for para in model.hard_fc1.parameters():
            para.requires_grad = True
        for para in model.hard_fc2.parameters():
            para.requires_grad = True
        optimizer_hard = SGD([{'params': model.hard_fc1.parameters()},
                              {'params': model.hard_fc2.parameters()}], lr=0.001)

        val_error, log_info, log_record = test_and_log(model, val_loader, test_loader,
                                                       device, 0, args.lr, 0, 0, log_record)
        write_log(log, log_info)

        for epoch in range(1, epochs + 1):
            _ = train(train_loader, model, optimizer, optimizer_hard, epsilon, lamda, lamda_ad, device)

            train_error, train_acc, _, _, _ = test(model, train_loader, device)
            val_error, log_info, log_record = test_and_log(model, val_loader, test_loader,
                                                           device, epoch, args.lr, train_error, train_acc,
                                                           log_record)
            write_log(log, log_info)

        log_record['mean acc'] = round(np.mean(log_record['test accs'][-10:]), 3)
        write_log(log, '')

        log_dict['record'].append(log_record)
        write_json(log_dict, log_json_path)
