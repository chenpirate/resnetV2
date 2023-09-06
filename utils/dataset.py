'''
Author: chenpirate chensy293@mail2.sysu.edu.cn
Date: 2023-02-21 11:10:59
LastEditors: chenpirate chensy293@mail2.sysu.edu.cn
LastEditTime: 2023-09-05 16:52:15
FilePath: /resnetV2/utils/dataset.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from os import cpu_count
import json
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader


# 自定义序列化方式
class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()


class HrrpDataset(Dataset):
    def __init__(self, path):
        self.xy = pd.read_csv(path, header=None)
        xy = self.xy
        x = xy.iloc[:, :-1].values
        x = x_norm(x)
        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x.float(), dim=1)
        label = xy.iloc[:, -1]
        enc = LabelEncoder()  # 获取一个LabelEncoder
        # labels = ['A330', 'A350', 'ARJ21', 'BY737', 'BY777', 'BY787', 'A321', 'A320', 'CRJ21']
        # labels = ['Prop1', 'Prop3', 'Prop4', 'Y20']
        labels = [1, 3, 5]
        enc = enc.fit(labels)
        dict_list = {}
        for cl in enc.classes_:
            dict_list.update({cl: int(enc.transform([cl])[0])})
        res = dict(zip(dict_list.values(), dict_list.keys()))
        with open("class_indices.json", "w", encoding='gbk') as f:
            json.dump(res, f, indent=2, sort_keys=True, ensure_ascii=False, cls=Encoder)
        self.y = enc.transform(label)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


def x_norm(x):
    x_norm = np.linalg.norm(x, ord=None, axis=1, keepdims=False).reshape(-1, 1)
    x = x / x_norm
    return x


def my_dataloader(root, shuffle=True, drop_last=True, batch_size=1, nw=cpu_count()):
    dataset = HrrpDataset(root)
    datasize = len(dataset)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=nw, shuffle=shuffle,
                             drop_last=drop_last)
    return data_loader, datasize


if __name__ == '__main__':
    _, train_size = my_dataloader('/home/private/hrrp/NanHu/20230419/test/test.csv', batch_size=32,
                                  nw=16)
    print("using {} HRRP datas for training.".format(train_size))
