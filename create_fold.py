import pandas as pd
import os
from sklearn.model_selection import KFold
import random
import pickle
import numpy as np
import math


def random_dic(dicts):
    # random.seed(2024)
    random.seed(2)
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    random.shuffle(dict_key_ls)
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic

#data_label {'文件名':1, '文件名':0}
def create_fold(data_label:dict, nfold=5, val_ratio=0.2, save_dir='create_csv/fold_io'):
    data = list(data_label.keys())
    nfold = KFold(n_splits=nfold, shuffle=True)
    for i, (train, test) in enumerate(nfold.split(data)):
        random.shuffle(train)
        random.shuffle(train)
        train_len = len(train)
        val_len = int(train_len * val_ratio)
        train_len = train_len - val_len
        val = train[train_len:]
        val = [data[_] for _ in val]
        train = train[:train_len]
        train = [data[_] for _ in train]
        test = [data[_] for _ in test]
        train_label = [int(float(data_label[_])) for _ in train]
        val_label = [int(float(data_label[_])) for _ in val]
        test_label = [int(float(data_label[_])) for _ in test]
        df1 = pd.DataFrame({'train':train, 'train_label':train_label}).astype(str)
        df2 = pd.DataFrame({'val':val, 'val_label':val_label}).astype(str)
        df3 = pd.DataFrame({'test':test, 'test_label':test_label}).astype(str)
        df = pd.concat([df1, df2, df3], axis=1)
        df.to_csv(f'{save_dir}/splits_{i}.csv', index=True)



if __name__ == '__main__':
# 随机打乱数据，分配数据，参数有 val_ratio， nfold， save_dir
    with open('all_PR.pkl', 'rb') as f:
        data = pickle.load(f)

    create_fold(data, save_dir='splits/5foldcv/PR')






