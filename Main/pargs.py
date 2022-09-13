import argparse
import os.path as osp

dirname = osp.dirname(osp.abspath(__file__))


def pargs():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Weibo')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--bert_path', type=str, default=osp.join(dirname, '..', 'Model', 'bert-base-chinese'))

    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)

    # 0.1/0.4
    parser.add_argument('--droprate', type=float, default=0.4)
    parser.add_argument('--join_source', type=str2bool, default=True)
    parser.add_argument('--mask_source', type=str2bool, default=True)
    parser.add_argument('--drop_mask_rate', type=float, default=0.15)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hid_feats', type=int, default=128)
    parser.add_argument('--out_feats', type=int, default=128)
    # 0.3/0.6
    parser.add_argument('--t', type=float, default=0.6)
    # [0.5, 0.3, 0.2]/[0.7, 0.2, 0.1]
    parser.add_argument('--probabilities', type=list, default=[0.7, 0.2, 0.1])

    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--epsilon', type=int, default=3)
    parser.add_argument('--lamda', type=float, default=0.001)
    parser.add_argument('--lamda_ad', type=float, default=0.001)

    parser.add_argument('--k', type=int, default=10000)

    args = parser.parse_args()
    return args

