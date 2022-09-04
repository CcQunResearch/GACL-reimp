# -*- coding: utf-8 -*-
# @Time    : 2022/9/2 20:05
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : model.py
# @Software: PyCharm
# @Note    :
import sys, os

sys.path.append(os.getcwd())
import torch as th
import torch.nn as nn
from torch_scatter import scatter_mean
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class hard_fc(th.nn.Module):
    def __init__(self, d_in, d_hid, DroPout=0):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(DroPout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.3, emb_name='hard_fc1.'):  # T15: epsilon = 0.2
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = th.norm(param.grad)
                if norm != 0 and not th.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='hard_fc1.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class GCN_Net(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats,t):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats, out_feats)
        self.fc = th.nn.Linear(2 * out_feats, 1)
        self.hard_fc1 = hard_fc(out_feats, out_feats)
        self.hard_fc2 = hard_fc(out_feats, out_feats)  # optional

        self.t = t

    def forward(self, data):
        init_x0, init_x, edge_index1, edge_index2 = data.x0, data.x, data.edge_index, data.edge_index2

        x1 = self.conv1(init_x0, edge_index1)
        x1 = F.relu(x1)
        x1 = self.conv2(x1, edge_index1)
        x1 = F.relu(x1)

        x1 = scatter_mean(x1, data.batch, dim=0)
        x1_g = x1
        x1 = self.hard_fc1(x1)
        x1_t = x1
        x1 = th.cat((x1_g, x1_t), 1)

        x2 = self.conv1(init_x, edge_index2)
        x2 = F.relu(x2)
        x2 = self.conv2(x2, edge_index2)
        x2 = F.relu(x2)
        x2 = scatter_mean(x2, data.batch, dim=0)
        x2_g = x2
        x2 = self.hard_fc1(x2)
        x2_t = x2
        x2 = th.cat((x2_g, x2_t), 1)
        x = th.cat((x1, x2), 0)
        y = th.cat((data.y1, data.y2), 0)

        x_T = x.t()
        dot_matrix = th.mm(x, x_T)
        x_norm = th.norm(x, p=2, dim=1)
        x_norm = x_norm.unsqueeze(1)
        norm_matrix = th.mm(x_norm, x_norm.t())

        cos_matrix = (dot_matrix / norm_matrix) / self.t
        cos_matrix = th.exp(cos_matrix)
        diag = th.diag(cos_matrix)
        cos_matrix_diag = th.diag_embed(diag)
        cos_matrix = cos_matrix - cos_matrix_diag
        y_matrix_T = y.expand(len(y), len(y))
        y_matrix = y_matrix_T.t()
        y_matrix = th.ne(y_matrix, y_matrix_T).float()
        # y_matrix_list = y_matrix.chunk(3, dim=0)
        # y_matrix = y_matrix_list[0]
        neg_matrix = cos_matrix * y_matrix
        neg_matrix_list = neg_matrix.chunk(2, dim=0)
        # neg_matrix = neg_matrix_list[0]
        pos_y_matrix = y_matrix * (-1) + 1
        pos_matrix_list = (cos_matrix * pos_y_matrix).chunk(2, dim=0)
        # print('cos_matrix: ', cos_matrix.shape, cos_matrix)
        # print('pos_y_matrix: ', pos_y_matrix.shape, pos_y_matrix)
        pos_matrix = pos_matrix_list[0]
        # print('pos shape: ', pos_matrix.shape, pos_matrix)
        neg_matrix = (th.sum(neg_matrix, dim=1)).unsqueeze(1)
        sum_neg_matrix_list = neg_matrix.chunk(2, dim=0)
        p1_neg_matrix = sum_neg_matrix_list[0]
        p2_neg_matrix = sum_neg_matrix_list[1]
        neg_matrix = p1_neg_matrix
        # print('neg shape: ', neg_matrix.shape)
        div = pos_matrix / neg_matrix
        div = (th.sum(div, dim=1)).unsqueeze(1)
        div = div / data.num_graphs
        log = th.log(div)
        SUM = th.sum(log)
        cl_loss = -SUM

        x = self.fc(x)
        x = F.sigmoid(x).view(-1)

        return x, cl_loss, y
