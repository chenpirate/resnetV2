'''
Author: chenpirate chensy293@mail2.sysu.edu.cn
Date: 2023-09-05 22:33:25
LastEditors: chenpirate chensy293@mail2.sysu.edu.cn
LastEditTime: 2023-09-06 15:42:31
FilePath: /resnetV2/utils/save_correctly_classified_output.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''


import json
import sys
from pandas import pd
from utils.exchange_json_key_value import exchange_json_key_value

from utils.norm import x_norm
sys.path.append('.')
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
def compute_distances(unknown, centroids):
    '''
    输入一个未知样本，所有已知类的质心，计算这个未知样本到各个已知类样本的质心距离
    返回这个距离List
    :param unknown: 未知样本
    :param centroids: 所有已知类的质心
    :return: 未知样本到各个已知类样本的质心距离
    '''
    distances = []
    for centroid in centroids:
        distances.append(compute_distance(unknown, centroid))
    return distances

# 修正得分函数
# 输入：未知样本到各个类质心的距离映射后概率tensor，未知样本的特征向量，这两个样本维度长度相同
# 输出：修正后的未知样本的特征向量
# 修正逻辑：
# 修正后的未知样本的特征向量=未知样本的特征向量×（1-概率tensor）
def correct_score(distances, unknown_feature, probabilities):
    '''
    修正得分函数
    :param distances: 未知样本到各个类质心的距离映射后概率tensor
    :param unknown_feature: 未知样本的特征向量
    :param probabilities: 未知样本到各个类质心的距离映射后概率tensor
    :return: 修正后的未知样本的特征向量
    '''
    return unknown_feature * (1 - probabilities)


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
    for i in range(len(unknown_feature)):
        score += unknown_feature[i] * (1 - probabilities[i])
    return score


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
    score_exp = torch.exp(score)
    score_sum = torch.sum(score_exp)
    probabilities = score_exp / score_sum
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


def get_centroids_and_cdfs(model, data_path, json_label_path):
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
        pred, output = save_correctly_classified_output(model, x[i], class_indict[label], device)
        if pred != -1:
            features[pred].append(output)
    
    # 计算每个类别的质心和每个样本到质心的距离
    for key, value in features.items():
        centroids[key]=compute_centroid(value)
        dists[key] = [compute_distance(output, centroids[key]) for output in value]

    # 利用距离拟合每一类的pdf和cdf
    cdfs = None

    return cdfs


# 矫正未知样本每类的得分并添加未知类的得分
def compute_correctly_classified_output(model, hrrp, centroids, cdfs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hrrp = x_norm(hrrp)
    _, output = predict(hrrp, model, device)
    distances = compute_distances(output, centroids)

    # 利用cdfs得到每个类别的概率
    # TODO
    probabilities=None

    # 修正已知类得分
    correct_feature = correct_score(distances, output, probabilities)

    # 未知类得分
    unknown = score_unknown(output, probabilities)

    # 拼接两类
    score = combine_score(correct_feature, unknown)
    return score
    
def recognition(score, json_label_path):
    # 获得已知类和未知类的概率
    class_index, prob = softmax_function(score)
    # 识别结果
    if is_unknown(prob, class_index, 0.85):
        print(f"拒判   属于库外概率为：{prob}")
    
    # labe_json的key，value对调
    label_json = open(json_label_path, 'r')
    label_json = json.load(label_json)
    exchange_json_key_value(label_json)

    print(f"识别结果为{label_json[class_index]}   属于库外概率为：{prob}")
    

def main(model, model_weight_path, data_path, json_label_path, unknow_targets):
    # 加载网络
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    # 获得cdfs and centroids
    cdfs, centroids = get_centroids_and_cdfs(model, data_path, json_label_path)

    # 修正得分并拼接未知类得分
    correct_score = compute_correctly_classified_output(model, unknow_targets, centroids, cdfs)

    # 识别
    recognition(correct_score, json_label_path)


if __name__ == "__main__":
    model = resnet18
    model_weight_path = ""
    data_path = ""
    json_label_path = ""
    unknown_target = None

    main(model, model_weight_path, data_path, json_label_path, unknown_target)

    