'''
Author: chenpirate chensy293@mail2.sysu.edu.cn
Date: 2023-09-06 16:03:26
LastEditors: chenpirate chensy293@mail2.sysu.edu.cn
LastEditTime: 2023-09-12 17:23:19
FilePath: /resnetV2/reject.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from utils.OpenMax import *


def main(model, model_weight_path, data_path, json_label_path, unknow_target_row_data, centroids_path, unit_measures_cdf_pairs_path):
    # 加载网络
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    get_probabilitys = GetProbalitys()
    centroids = None
    unit_measures_cdf_pairs = None
    try:
        centroids_json_file = open(centroids_path, 'r')
        centroids = json.load(centroids_json_file)
        for key, value in centroids.items():
            centroids[key] = torch.tensor(value)

        unit_measures_cdf_pairs = open(unit_measures_cdf_pairs_path, 'r')
        unit_measures_cdf_pairs=json.load(unit_measures_cdf_pairs)
        for key, value in unit_measures_cdf_pairs.items():
            value[0] = torch.tensor(value[0])
            unit_measures_cdf_pairs[key] = value
    except:
        # 获得库内的类别距离和各类质心
        known_distances, centroids = get_dists_and_centroids(model, data_path, json_label_path)
        
        centroids_json = {}
        for key, value in centroids.items():
            centroids_json[key] = value.tolist()
        with open(centroids_path, "w") as f:
            json.dump(centroids_json, f)
            
        # get known target cdf and unknow target probabilitys
        get_probabilitys = GetProbalitys()
        unit_measures_cdf_pairs = get_probabilitys.get_unit_measures_cdf_pairs(known_distances)
        unit_measures_cdf_pairs_json = {}
        for key, value in unit_measures_cdf_pairs.items():
            # print(type(value))
            value[0] = value[0].tolist()
            unit_measures_cdf_pairs_json[key] = value

        with open(unit_measures_cdf_pairs_path, "w") as f:
            json.dump(unit_measures_cdf_pairs_json, f)
            
    _, unknown_feature = predict(unknow_target_row_data, model, device)

    # 未知类与其他已知类质心的距离
    dists_with_each_known_centroid = compute_distances(unknown_feature, centroids, device)
    unknow_target_probabilitys = get_probabilitys.get_probabilitys(dists_with_each_known_centroid, unit_measures_cdf_pairs)

    # 修正得分并拼接未知类得分
    correct_score = compute_correctly_classified_output(unknown_feature, unknow_target_probabilitys)
    # 识别
    recognition(correct_score, json_label_path, 0.6)


if __name__ == "__main__":
    model = resnet18(6)
    model_weight_path = "weight/9_12_best.pth"
    data_path = "data/9.12/testdata_013489.csv"
    json_label_path = "class_indices.json"

    unknown_targets_path = 'data/9.12/库外目标_5.csv'
    df = pd.read_csv(data_path)
    for i in range(len(df)):
        unknown_target = df.iloc[i, :-1].values
        unknown_target = x_norm(unknown_target)
        print(df.iloc[i, -1])
        main(model, model_weight_path, data_path, json_label_path, unknown_target, centroids_path="centroids.json", unit_measures_cdf_pairs_path="unit_measures_cdf_pairs.json")
