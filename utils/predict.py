'''
Author: chenpirate chensy293@mail2.sysu.edu.cn
Date: 2023-02-21 11:10:59
LastEditors: chenpirate chensy293@mail2.sysu.edu.cn
LastEditTime: 2023-09-12 13:43:59
FilePath: /resnetV2/utils/predict.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import json
import torch
import pandas as pd
import time
import numpy as np
from model import resnet18
from utils.norm import x_norm

# 有一个训练好的分类网络，并加载好了权重
# 分类网络的输出只是最后一层全连接层的输出，因此需要经过softmax函数，并取最大的概率索引，作为预测类别
# 使用这个网络，对一条信号进行分类，返回预测类别以及分类网络的输出
# 实现这个函数
def predict(hrrp, model, device="cpu"):
    model.eval()
    with torch.no_grad():
        hrrp = torch.from_numpy(hrrp).float().to(device)
        if hrrp.ndim != 3:
            hrrp = hrrp.reshape(1, 1, hrrp.size()[-1])
        output = model(hrrp)
        prob = torch.nn.functional.softmax(output, dim=1)
        prob = prob.cpu().numpy()
        pred = prob.argmax(axis=1)
        return pred[0], output

def main():
    # 加载网络
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet18(num_classes=10).to(device)
    model_weight_path = "./model/resnet18_epoch_100.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    # 加载数据并归一化
    data_path = "./data/test.csv"
    df = pd.read_csv(data_path)
    x = df.iloc[:, :-1].values
    x = x_norm(x)

    # 加载类别json
    json_label_path = "./data/label_list.json"
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    # 预测
    pred, output = predict(x[0], model, device)
    print(pred, output)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("循环运行时间:%.2f秒" % (end - start))