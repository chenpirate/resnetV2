'''
Author: chenpirate chensy293@mail2.sysu.edu.cn
Date: 2023-09-06 16:03:26
LastEditors: chenpirate chensy293@mail2.sysu.edu.cn
LastEditTime: 2023-09-12 09:34:31
FilePath: /resnetV2/reject.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from utils.OpenMax import *


def main(model, model_weight_path, data_path, json_label_path, unknow_target_row_data):
    # 加载网络
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    # 获得库内的类别距离和各类质心
    known_distances, centroids = get_dists_and_centroids(model, data_path, json_label_path)

    # get known target cdf and unknow target probabilitys
    get_probabilitys = GetProbalitys()
    unit_measures_cdf_pairs = get_probabilitys.get_unit_measures_cdf_pairs(known_distances)

    unknown_feature = predict(unknow_target_row_data, model, device)

    # 未知类与其他已知类质心的距离
    dists_with_each_known_centroid = compute_distances(unknown_feature, centroids)
    unkonw_target_probabilitys = get_probabilitys.get_probabilitys(dists_with_each_known_centroid, unit_measures_cdf_pairs)


    # 修正得分并拼接未知类得分
    correct_score = compute_correctly_classified_output(model, unknown_feature, centroids, unkonw_target_probabilitys)

    # 识别
    recognition(correct_score, json_label_path)


if __name__ == "__main__":
    model = resnet18
    model_weight_path = ""
    data_path = ""
    json_label_path = ""
    unknown_target = None

    main(model, model_weight_path, data_path, json_label_path, unknown_target)
