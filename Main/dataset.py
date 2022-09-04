# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 18:59
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : dataset.py
# @Software: PyCharm
# @Note    :
import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from Main.utils import clean_comment
from Main.bert import get_sent_embedding


def random_pick(list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


class WeiboDataset(Dataset):
    def __init__(self, root, bert_tokenizer, bert_model, probabilities, clean=True, droprate=0.0, join_source=False, mask_source=False,
                 drop_mask_rate=0.0):
        self.root = root
        self.raw_dir = os.path.join(self.root, 'raw')
        self.bert_tokenizer = bert_tokenizer
        self.bert_model = bert_model
        self.probabilities = probabilities
        self.clean = clean
        self.droprate = droprate
        self.join_source = join_source
        self.mask_source = mask_source
        self.drop_mask_rate = drop_mask_rate
        self.data_list = self.process()

    def process(self):
        data_list = []
        raw_file_names = os.listdir(self.raw_dir)

        if self.clean:
            limit_num = 600
            pass_comment = ['', '转发微博', '转发微博。', '轉發微博', '轉發微博。']
            for filename in raw_file_names:
                y = []
                row = []
                col = []
                filepath = os.path.join(self.raw_dir, filename)
                post = json.load(open(filepath, 'r', encoding='utf-8'))
                x0 = get_sent_embedding(post['source']['content'], '', self.bert_tokenizer, self.bert_model).view(1, -1)
                x = get_sent_embedding(post['source']['content'], '', self.bert_tokenizer, self.bert_model,
                                       drop_mask_rate=self.drop_mask_rate).view(1, -1)
                y.append(post['source']['label'])
                pass_num = 0
                id_to_index = {}
                for i, comment in enumerate(post['comment']):
                    if i == limit_num:
                        break
                    id_to_index[comment['comment id']] = i
                for i, comment in enumerate(post['comment']):
                    if i == limit_num:
                        break
                    if comment['content'] in pass_comment and comment['children'] == []:
                        pass_num += 1
                        continue
                    post['comment'][i]['comment id'] -= pass_num
                for i, comment in enumerate(post['comment']):
                    if i == limit_num:
                        break
                    if comment['content'] in pass_comment and comment['children'] == []:
                        continue
                    commnet_embedding0 = get_sent_embedding(clean_comment(comment['content']),
                                                            post['source']['content'],
                                                            self.bert_tokenizer, self.bert_model,
                                                            join_sents=self.join_source).view(1, -1)
                    commnet_embedding = get_sent_embedding(clean_comment(comment['content']), post['source']['content'],
                                                           self.bert_tokenizer, self.bert_model,
                                                           join_sents=self.join_source, mask_join_sent=self.mask_source,
                                                           drop_mask_rate=self.drop_mask_rate).view(1, -1)
                    x0 = torch.cat(
                        [x0, commnet_embedding0], 0)
                    x = torch.cat(
                        [x, commnet_embedding], 0)
                    if comment['parent'] == -1:
                        row.append(0)
                    else:
                        row.append(post['comment'][id_to_index[comment['parent']]]['comment id'] + 1)
                    col.append(comment['comment id'] + 1)

                burow = col
                bucol = row
                new_row = row + burow
                new_col = col + bucol
                new_edgeindex = [new_row, new_col]

                # ==================================- dropping + adding + misplacing -===================================#

                choose_list = [1, 2, 3]  # 1-drop 2-add 3-misplace
                choose_num = random_pick(choose_list, self.probabilities)

                if self.droprate > 0:
                    if choose_num == 1:
                        length = len(new_row)
                        poslist = random.sample(range(length), int(length * (1 - self.droprate)))
                        poslist = sorted(poslist)
                        new_row2 = list(np.array(new_row)[poslist])
                        new_col2 = list(np.array(new_col)[poslist])
                        new_edgeindex2 = [new_row2, new_col2]
                    elif choose_num == 2:
                        length = len(list(set(sorted(new_row))))
                        add_row = random.sample(range(length), int(length * self.droprate))
                        add_col = random.sample(range(length), int(length * self.droprate))
                        new_row2 = new_row + add_row + add_col
                        new_col2 = new_col + add_col + add_row
                        new_edgeindex2 = [new_row2, new_col2]
                    elif choose_num == 3:
                        length = len(row)
                        mis_index_list = random.sample(range(length), int(length * self.droprate))
                        Sort_len = len(list(set(sorted(new_row))))
                        if Sort_len > int(length * self.droprate):
                            mis_value_list = random.sample(range(Sort_len), int(length * self.droprate))
                            for i, item in enumerate(row):
                                for mis_i, mis_item in enumerate(mis_index_list):
                                    if i == mis_item and mis_value_list[mis_i] != item:
                                        row[i] = mis_value_list[mis_i]
                            new_row2 = row + col
                            new_col2 = col + row
                            new_edgeindex2 = [new_row2, new_col2]
                        else:
                            length = len(new_row)
                            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
                            poslist = sorted(poslist)
                            new_row2 = list(np.array(new_row)[poslist])
                            new_col2 = list(np.array(new_col)[poslist])
                            new_edgeindex2 = [new_row2, new_col2]
                else:
                    new_edgeindex = [new_row, new_col]
                    new_edgeindex2 = [new_row, new_col]

                if self.droprate > 0:
                    if choose_num == 1:
                        x_length = x.shape[0]
                        mask_pos = random.sample(range(x_length), int(x_length * self.droprate))
                        mask_pos = torch.tensor(sorted(mask_pos))
                        if len(mask_pos.tolist()) != 0:
                            x[mask_pos] = torch.zeros(x.shape[1])

                one_data = Data(x0=x0, x=x, edge_index=torch.LongTensor(new_edgeindex),
                                edge_index2=torch.LongTensor(new_edgeindex2),
                                y1=torch.LongTensor([y]), y2=torch.LongTensor([y]))
                data_list.append(one_data)
        else:
            for filename in raw_file_names:
                y = []
                row = []
                col = []
                filepath = os.path.join(self.raw_dir, filename)
                post = json.load(open(filepath, 'r', encoding='utf-8'))
                x0 = get_sent_embedding(post['source']['content'], '', self.bert_tokenizer, self.bert_model).view(1, -1)
                x = get_sent_embedding(post['source']['content'], '', self.bert_tokenizer, self.bert_model,
                                       drop_mask_rate=self.drop_mask_rate).view(1, -1)
                y.append(post['source']['label'])
                for i, comment in enumerate(post['comment']):
                    commnet_embedding0 = get_sent_embedding(clean_comment(comment['content']),
                                                            post['source']['content'],
                                                            self.bert_tokenizer, self.bert_model,
                                                            join_sents=self.join_source).view(1, -1)
                    commnet_embedding = get_sent_embedding(clean_comment(comment['content']), post['source']['content'],
                                                           self.bert_tokenizer, self.bert_model,
                                                           join_sents=self.join_source, mask_join_sent=self.mask_source,
                                                           drop_mask_rate=self.drop_mask_rate).view(1, -1)
                    x0 = torch.cat(
                        [x0, commnet_embedding0], 0)
                    x = torch.cat(
                        [x, commnet_embedding], 0)
                    row.append(comment['parent'] + 1)
                    col.append(comment['comment id'] + 1)

                burow = col
                bucol = row
                new_row = row + burow
                new_col = col + bucol
                new_edgeindex = [new_row, new_col]

                # ==================================- dropping + adding + misplacing -===================================#

                choose_list = [1, 2, 3]  # 1-drop 2-add 3-misplace
                probabilities = [0.7, 0.2, 0.1]
                choose_num = random_pick(choose_list, probabilities)

                if self.droprate > 0:
                    if choose_num == 1:
                        length = len(new_row)
                        poslist = random.sample(range(length), int(length * (1 - self.droprate)))
                        poslist = sorted(poslist)
                        new_row2 = list(np.array(new_row)[poslist])
                        new_col2 = list(np.array(new_col)[poslist])
                        new_edgeindex2 = [new_row2, new_col2]
                    elif choose_num == 2:
                        length = len(list(set(sorted(new_row))))
                        add_row = random.sample(range(length), int(length * self.droprate))
                        add_col = random.sample(range(length), int(length * self.droprate))
                        new_row2 = new_row + add_row + add_col
                        new_col2 = new_col + add_col + add_row
                        new_edgeindex2 = [new_row2, new_col2]
                    elif choose_num == 3:
                        length = len(row)
                        mis_index_list = random.sample(range(length), int(length * self.droprate))
                        Sort_len = len(list(set(sorted(new_row))))
                        if Sort_len > int(length * self.droprate):
                            mis_value_list = random.sample(range(Sort_len), int(length * self.droprate))
                            for i, item in enumerate(row):
                                for mis_i, mis_item in enumerate(mis_index_list):
                                    if i == mis_item and mis_value_list[mis_i] != item:
                                        row[i] = mis_value_list[mis_i]
                            new_row2 = row + col
                            new_col2 = col + row
                            new_edgeindex2 = [new_row2, new_col2]
                        else:
                            length = len(new_row)
                            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
                            poslist = sorted(poslist)
                            new_row2 = list(np.array(new_row)[poslist])
                            new_col2 = list(np.array(new_col)[poslist])
                            new_edgeindex2 = [new_row2, new_col2]
                else:
                    new_edgeindex = [new_row, new_col]
                    new_edgeindex2 = [new_row, new_col]

                if self.droprate > 0:
                    if choose_num == 1:
                        x_length = x.shape[0]
                        mask_pos = random.sample(range(x_length), int(x_length * self.droprate))
                        mask_pos = torch.tensor(sorted(mask_pos))
                        if len(mask_pos.tolist()) != 0:
                            x[mask_pos] = torch.zeros(x.shape[1])

                one_data = Data(x0=x0, x=x, edge_index=torch.LongTensor(new_edgeindex),
                                edge_index2=torch.LongTensor(new_edgeindex2),
                                y1=torch.LongTensor([y]), y2=torch.LongTensor([y]))
                data_list.append(one_data)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


class WeiboDatasetTest(Dataset):
    def __init__(self, root, bert_tokenizer, bert_model, clean=True, join_source=False):
        self.root = root
        self.raw_dir = os.path.join(self.root, 'raw')
        self.bert_tokenizer = bert_tokenizer
        self.bert_model = bert_model
        self.clean = clean
        self.join_source = join_source
        self.data_list = self.process()

    def process(self):
        data_list = []
        raw_file_names = os.listdir(self.raw_dir)

        if self.clean:
            limit_num = 600
            pass_comment = ['', '转发微博', '转发微博。', '轉發微博', '轉發微博。']
            for filename in raw_file_names:
                y = []
                row = []
                col = []
                filepath = os.path.join(self.raw_dir, filename)
                post = json.load(open(filepath, 'r', encoding='utf-8'))
                x = get_sent_embedding(post['source']['content'], '', self.bert_tokenizer, self.bert_model).view(1, -1)
                y.append(post['source']['label'])
                pass_num = 0
                id_to_index = {}
                for i, comment in enumerate(post['comment']):
                    if i == limit_num:
                        break
                    id_to_index[comment['comment id']] = i
                for i, comment in enumerate(post['comment']):
                    if i == limit_num:
                        break
                    if comment['content'] in pass_comment and comment['children'] == []:
                        pass_num += 1
                        continue
                    post['comment'][i]['comment id'] -= pass_num
                for i, comment in enumerate(post['comment']):
                    if i == limit_num:
                        break
                    if comment['content'] in pass_comment and comment['children'] == []:
                        continue
                    commnet_embedding = get_sent_embedding(clean_comment(comment['content']), post['source']['content'],
                                                           self.bert_tokenizer, self.bert_model,
                                                           join_sents=self.join_source).view(1, -1)
                    x = torch.cat(
                        [x, commnet_embedding], 0)
                    if comment['parent'] == -1:
                        row.append(0)
                    else:
                        row.append(post['comment'][id_to_index[comment['parent']]]['comment id'] + 1)
                    col.append(comment['comment id'] + 1)

                burow = col
                bucol = row
                new_row = row + burow
                new_col = col + bucol
                new_edgeindex = [new_row, new_col]

                one_data = Data(x0=x, x=x, edge_index=torch.LongTensor(new_edgeindex),
                                edge_index2=torch.LongTensor(new_edgeindex),
                                y1=torch.LongTensor([y]), y2=torch.LongTensor([y]))
                data_list.append(one_data)
        else:
            for filename in raw_file_names:
                y = []
                row = []
                col = []
                filepath = os.path.join(self.raw_dir, filename)
                post = json.load(open(filepath, 'r', encoding='utf-8'))
                x = get_sent_embedding(post['source']['content'], '', self.bert_tokenizer, self.bert_model).view(1, -1)
                y.append(post['source']['label'])
                for i, comment in enumerate(post['comment']):
                    commnet_embedding = get_sent_embedding(clean_comment(comment['content']), post['source']['content'],
                                                           self.bert_tokenizer, self.bert_model,
                                                           join_sents=self.join_source).view(1, -1)
                    x = torch.cat(
                        [x, commnet_embedding], 0)
                    row.append(comment['parent'] + 1)
                    col.append(comment['comment id'] + 1)

                burow = col
                bucol = row
                new_row = row + burow
                new_col = col + bucol
                new_edgeindex = [new_row, new_col]

                one_data = Data(x0=x, x=x, edge_index=torch.LongTensor(new_edgeindex),
                                edge_index2=torch.LongTensor(new_edgeindex),
                                y1=torch.LongTensor([y]), y2=torch.LongTensor([y]))
                data_list.append(one_data)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]
