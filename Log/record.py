# -*- coding: utf-8 -*-
# @Time    : 2022/5/7 20:26
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : record.py
# @Software: PyCharm
# @Note    :
import os
import sys
import json
import math
import numpy as np

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, '..'))

cal_mean = -10

if __name__ == '__main__':
    log_dir_path = os.path.join(dirname, '..', 'Log')

    for filename in os.listdir(log_dir_path):
        if filename[-4:] == 'json':
            print(f'【{filename[:-5]}】')
            filepath = os.path.join(log_dir_path, filename)

            log = json.load(open(filepath, 'r', encoding='utf-8'))

            print('dataset', log['dataset'])
            print('runs', log['runs'])
            print('bert_path', log['bert_path'])

            print('droprate', log['droprate'])
            print('join_source', log['join_source'])
            print('mask_source', log['mask_source'])
            print('drop_mask_rate', log['drop_mask_rate'])

            print('batch_size', log['batch_size'])
            print('hid_feats', log['hid_feats'])
            print('out_feats', log['out_feats'])

            print('t', log['t'])
            print('probabilities', log['probabilities'])

            print('lr', log['lr'])
            print('epochs', log['epochs'])
            print('weight_decay', log['weight_decay'])
            print('epsilon', log['epsilon'])
            print('lamda', log['lamda'])
            print('lamda_ad', log['lamda_ad'])

            print('dataset', log['k'])

            acc_list = []
            for run in log['record']:
                # mean_acc = run['mean acc']
                mean_acc = round(np.mean(run['test accs'][cal_mean:]), 3)
                acc_list.append(mean_acc)

            mean = round(sum(acc_list) / len(acc_list), 3)
            sd = round(math.sqrt(sum([(x - mean) ** 2 for x in acc_list]) / len(acc_list)), 3)
            maxx = max(acc_list)
            print('test acc: {:.3f}±{:.3f}'.format(mean, sd))
            print('max acc: {:.3f}'.format(maxx))
            print()
