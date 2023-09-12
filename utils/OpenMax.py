'''
Author: chenpirate chensy293@mail2.sysu.edu.cn
Date: 2023-09-05 22:33:25
LastEditors: chenpirate chensy293@mail2.sysu.edu.cn
LastEditTime: 2023-09-12 16:54:14
FilePath: /resnetV2/utils/save_correctly_classified_output.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''


import json
import sys
import pandas as pd
from utils.exchange_json_key_value import exchange_json_key_value

from utils.norm import x_norm
sys.path.append('.')
import torch
import torch.nn as nn
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
    if pred == label:
        return pred, output
    return -1, -1


# 计算质心的函数，输入：一个outputs list，存放同一类别的每个样本的output tensor，输出：质心位置tensor
def compute_centroid(outputs):
    '''
    计算质心
    :param outputs: 一个outputs list，存放同一类别的每个样本的output tensor
    :return: 质心位置tensor
    '''
    return torch.mean(torch.stack(outputs), dim=0)

# 计算单个样本距离的函数，输入：样本的高维output tensor，相应类别的质心位置tensor，输出：样本与质心的距离
def compute_distance(output, centroid):
    '''
    计算距离
    :param output: 样本的高维output tensor
    :param centroid: 相应类别的质心位置tensor
    :return: 样本与质心的距离
    '''
    
    return torch.dist(output, centroid)

# 输入一个未知样本，所有已知类的质心，计算这个未知样本到各个已知类样本的质心距离
# 返回这个距离List
def compute_distances(unknown, centroids, device):
    '''
    输入一个未知样本，所有已知类的质心，计算这个未知样本到各个已知类样本的质心距离
    返回这个距离List
    :param unknown: 未知样本
    :param centroids: 所有已知类的质心
    :return: 未知样本到各个已知类样本的质心距离
    '''
    distances = []
    for _, centroid in centroids.items():
        distances.append(compute_distance(unknown, centroid.to(device)))
    return distances

def get_probabilities(unknow, centroids, know_distances):
    probabilities = []
    for i in range(len(centroids)):
        distance = compute_distance(unknow, centroids[i])
        # 遍历know_distances[i]，求小于distance的个数，计算占比
        probabilities.append(sum(1 for d in know_distances[i] if d < distance) / len(know_distances[i]))
    return probabilities


# 获取未知样本的概率
class GetProbalitys():
    def __init__(self) -> None:
        # self.target_map_unit_measures_cdf_pairs = {}
        pass

    def get_unit_measures_cdf_pairs(self, known_distances_dict:dict):
        '''
        :param known_distances_dict:已知类的网络输出与质心的距离
        '''

        target_map_unit_measures_cdf_pairs = {}
        for target, distances in known_distances_dict.items():
            # 对datas从小到大排列
            distances.sort()
            max_value = distances[-1]
            unit_measures = max_value/100
            cdf = []
            for i in range(1, 100):
                count = 0
                for distance in distances:
                    if distance>i*unit_measures: break
                    count += 1
                # print(f"i:{i} \t count:{count}")
                cdf.append(count/len(distances))
            # 反转cdf
            cdf.reverse()
            target_map_unit_measures_cdf_pairs[target] = [unit_measures, cdf]
        return target_map_unit_measures_cdf_pairs


    def get_probabilitys(self, dists_with_each_known_centroid, unit_measures_cdf_pairs)->dict:
        # 获取未知样本的概率属于库内各类的概率
        probabilitys = {}
        for target, dist in enumerate(dists_with_each_known_centroid):
            unit_measures, cdf = unit_measures_cdf_pairs[str(target)]
            idx = int(dist/unit_measures)
            try:
                probabilitys[target] = cdf[idx]   
            except IndexError:
                probabilitys[target] = cdf[-1]
        return probabilitys


# 修正得分函数
# 输入：未知样本到各个类质心的距离映射后概率tensor，未知样本的特征向量，这两个样本维度长度相同
# 输出：修正后的未知样本的特征向量
# 修正逻辑：
# 修正后的未知样本的特征向量=未知样本的特征向量×（1-概率tensor）
def correct_score(unknown_feature, unkonw_target_probabilitys):
    '''
    修正得分函数
    :param distances: 未知样本到各个类质心的距离映射后概率tensor
    :param unknown_feature: 未知样本的特征向量
    :param unkonw_target_probabilitys: 未知样本到各个类质心的距离映射后概率tensor
    :return: 修正后的未知样本的特征向量
    '''
    score = []
    for key, value in unkonw_target_probabilitys.items():
        score.append(unknown_feature[0][key]*(value))

    return score


# 未知类的得分
# 输入：未知样本到各个类质心的距离映射后概率tensor，未知样本的特征向量，这两个样本维度长度相同
# 输出：未知类的得分Score_unknow
# 修正逻辑：Score_unknow，Score_unknown=Score1*(1-w1)+Score2*(1-w2)+...+ScoreK*(1-wk)
# wi=1-对应类的概率
def score_unknown(unknown_feature, probabilities):
    '''
    未知类的得分
    :param distances: 未知样本到各个类质心的距离映射后概率tensor
    :param unknown_feature: 未知样本的特征向量
    :param probabilities: 未知样本到各个类质心的距离映射后概率tensor
    :return: 未知类的得分Score_unknow
    '''
    score = 0
    # probabilities = probabilities.tolist()
    unknown_feature = unknown_feature.tolist()
    for i in range(len(unknown_feature[0])):
        score += unknown_feature[0][i] * (1-probabilities[i])
    return torch.tensor(score)


# 组和成新的得分向量
# 输入修正后的得分向量，未知类的得分
# 输出新的得分向量
# 逻辑：在修正后的得分向量末尾添加未知类的得分
def combine_score(corrected_score, score_unknown):
    '''
    组和成新的得分向量
    :param corrected_score: 修正后的得分向量
    :param score_unknown: 未知类的得分
    :return: 新的得分向量
    '''
    return torch.cat((corrected_score, score_unknown.reshape(1)), dim=0)

# 利用SoftMax函数计算概率，返回最大概率值和对应的索引
# 输入：得分向量
# 输出：最大概率值，最大概率值对应的索引
def softmax_function(score):
    '''
    利用SoftMax函数计算概率，返回最大概率值和对应的索引
    :param score: 得分向量
    :return: 最大概率值，最大概率值对应的索引
    '''
    probabilities = nn.Softmax(dim=0)(score)
    return torch.argmax(probabilities), torch.max(probabilities)

# 判断是否为未知类，是则输出未知类，概率值
# 否输出类别，概率值
# 输入：概率，识别的类别
# 输出：类别，概率值
# 输入类别为未知类，或者分类概率小于某个阈值，识别为未知类
def is_unknown(probabilitie, class_index, threadhold):
    '''
    判断是否为未知类，是则输出未知类，概率值
    否输出类别，概率值
    输入：概率，识别的类别
    输出：类别，概率值
    输入类别为未知类，或者分类概率小于某个阈值，识别为未知类
    :param probabilities: 概率
    :param class_index: 识别的类别
    :return: 类别，概率值
    '''
    if class_index == -1 or probabilitie < threadhold:
        return True
    return False


def get_dists_and_centroids(model, data_path, json_label_path):
    # 加载网络
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = resnet18(num_classes=10).to(device)
    # model_weight_path = "./model/resnet18_epoch_100.pth"
    # model.load_state_dict(torch.load(model_weight_path, map_location=device))

    # 加载数据并归一化
    df = pd.read_csv(data_path)
    x = df.iloc[:, :-1].values
    x = x_norm(x)

    labels = df.iloc[:, -1]

    # 加载类别json
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    # 对class_indict遍历，得到一个features = {key, []}的dict，key为class_indict的value
    features = {}
    centroids = {}
    dists = {}
    for _, value in class_indict.items():
        features[value] = []
        centroids[value] = None
        dists[value] = []

    for i in range(len(x)):
        # 预测
        label = labels[i]
        # print(f"label:{label}")
        pred, output = save_correctly_classified_output(model, x[i], class_indict[str(label)], device)
        if pred != -1:
            features[pred].append(output)
    
    # 计算每个类别的质心和每个样本到质心的距离
    for key, value in features.items():
        centroids[key]=compute_centroid(value)
        dists[key] = [compute_distance(output, centroids[key]) for output in value]

    return dists, centroids


# 矫正未知样本每类的得分并添加未知类的得分
def compute_correctly_classified_output(hrrp, probabilities):

    # 修正已知类得分
    correct_feature = torch.Tensor(correct_score(hrrp, probabilities))

    # 未知类得分
    unknown = score_unknown(hrrp, probabilities)

    # 拼接
    score = combine_score(correct_feature, unknown)
    return score
    

def recognition(score, json_label_path, threahold):
    # 获得已知类和未知类的概率
    class_index, prob = softmax_function(score)
    # 识别结果
    # if is_unknown(prob, class_index, 0.70):
    #     print(f"拒判   属于库外概率为：{prob}")
    #     return 
    
    # labe_json的key，value对调
    label_json = open(json_label_path, 'r')
    label_json = json.load(label_json)
    label_json = exchange_json_key_value(label_json)

    if class_index>=len(label_json):
        print(f"拒判   属于库外概率为：{prob}")
    elif  prob<threahold:
        print(f"拒判   属于库外概率为：{1-prob}")
    else:
        print(f"识别结果为{label_json[int(class_index)]}   置信度为：{prob}")