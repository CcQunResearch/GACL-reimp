# 一、命令

进入文件夹：

```shell script
cd /home/ccq/Research/GACL-reimp/Main
```

运行程序：

```shell script
nohup python main.py --gpu 0 &
```

# 二、实验

1. Weibo

- 实验结果

| id | dataset | vec size | bert version | droprate | join_source | mask_source | drop_mask_rate | t | probabilities | hid_feats | out_feats | weight decay | batch size | epsilon | lamda | lamda_ad | test acc | max acc |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 1 | Weibo | 768 | bert-base-chinese | **0.1** | True | True | 0.15 | **0.3** | **[0.7, 0.2, 0.1]** | 128 | 128 | 0.0001 | 128 | 3 | 0.001 | 0.001 | 0.935±0.004 | 0.941 | 
| 1 | Weibo | 768 | bert-base-chinese | **0.1** | True | True | 0.15 | **0.6** | **[0.7, 0.2, 0.1]** | 128 | 128 | 0.0001 | 128 | 3 | 0.001 | 0.001 | 0.933±0.008 | 0.943 | 
| 1 | Weibo | 768 | bert-base-chinese | **0.1** | True | True | 0.15 | **0.3** | **[0.5, 0.3, 0.2]** | 128 | 128 | 0.0001 | 128 | 3 | 0.001 | 0.001 | 0.935±0.009 | 0.951 | 
| 1 | Weibo | 768 | bert-base-chinese | **0.1** | True | True | 0.15 | **0.6** | **[0.5, 0.3, 0.2]** | 128 | 128 | 0.0001 | 128 | 3 | 0.001 | 0.001 | 0.938±0.006 | 0.944 | 
| 1 | Weibo | 768 | bert-base-chinese | **0.4** | True | True | 0.15 | **0.3** | **[0.7, 0.2, 0.1]** | 128 | 128 | 0.0001 | 128 | 3 | 0.001 | 0.001 | 0.935±0.005 | 0.944 | 
| 1 | Weibo | 768 | bert-base-chinese | **0.4** | True | True | 0.15 | **0.6** | **[0.7, 0.2, 0.1]** | 128 | 128 | 0.0001 | 128 | 3 | 0.001 | 0.001 | 0.940±0.006 | 0.951 | 
| 1 | Weibo | 768 | bert-base-chinese | **0.4** | True | True | 0.15 | **0.3** | **[0.5, 0.3, 0.2]** | 128 | 128 | 0.0001 | 128 | 3 | 0.001 | 0.001 | 0.934±0.005 | 0.944 | 
| 1 | Weibo | 768 | bert-base-chinese | **0.4** | True | True | 0.15 | **0.6** | **[0.5, 0.3, 0.2]** | 128 | 128 | 0.0001 | 128 | 3 | 0.001 | 0.001 | 0.936±0.005 | 0.948 | 
