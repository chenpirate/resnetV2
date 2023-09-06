'''
Author: chenpirate chensy293@mail2.sysu.edu.cn
Date: 2023-09-05 22:33:25
LastEditors: chenpirate chensy293@mail2.sysu.edu.cn
LastEditTime: 2023-09-06 11:04:42
FilePath: /resnetV2/utils/save_correctly_classified_output.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''


import sys
import torch
import tqdm
from utils.predict import predict
from model import resnet18


def save_correctly_classified_output(model, data, label, device):
    '''
    根据输入的数据和模型，对预测结果进行判断，并返回正确预测结果和输出
    :param model: 模型
    :param data: 数据
    :param label: 标签
    :param device: 设备
    :return: pred: 预测结果，output: 输出
    '''
    pred, output = predict(data, model, device)
    pred_classes = torch.max(pred, dim=1)[1]
    if pred_classes == label:
        return pred, output
    return None

# 计算质心的函数，输入：同一类别的高维outputs tensor，输出：质心位置tensor
def compute_centroid(outputs):
    '''
    计算质心
    :param outputs: 同一类别的高维outputs tensor
    :return: 质心位置tensor
    '''
    return torch.mean(outputs, dim=0)

# 计算距离的函数，输入：样本的高维output tensor，相应类别的质心位置tensor，输出：样本与质心的距离
def compute_distance(output, centroid):
    '''
    计算距离
    :param output: 样本的高维output tensor
    :param centroid: 相应类别的质心位置tensor
    :return: 样本与质心的距离
    '''
    return torch.dist(output, centroid)

# 计算距离的累计分布函数，输入：所有样本与质心的距离，数据格式list，每个元素都是单个样本与质心的距离，输出：距离的累计分布函数
def compute_cdf(distances:list):
    '''
    计算距离的累计分布函数
    :param distances: 所有样本与质心的距离，数据格式list，每个元素都是单个样本与质心的距离
    :return: 距离的累计分布函数
    '''
    distances = torch.tensor(distances)
    distances_sorted, _ = torch.sort(distances)
    cdf = torch.cumsum(distances_sorted, dim=0)
    cdf = cdf / cdf[-1]
    return cdf


