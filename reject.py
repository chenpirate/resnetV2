'''
Author: chenpirate chensy293@mail2.sysu.edu.cn
Date: 2023-09-06 16:03:26
LastEditors: chenpirate chensy293@mail2.sysu.edu.cn
LastEditTime: 2023-09-06 16:03:48
FilePath: /resnetV2/reject.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from utils.OpenMax import *


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
