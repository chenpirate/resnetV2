import os
import json
import torch
import pandas as pd
import time
from contrative.restnet34 import resnet34


def main():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("using {} device.".format(device))

    # read class_indict
    json_path = './real_class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)  # 判断路径是否有此文件存在

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)  # 将json_file文件从字典转换为字符串

    # 读取数据
    df = pd.read_csv("test.csv", header=None)
    data = df.iloc[0, :-1].values.reshape(1, 1, 1024)

    # 转换成tensor
    data = torch.from_numpy(data)

    # 创建模型
    model = resnet34().to(device)

    # 载入模型权重
    weights_path = "./CNN_1D.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(data.float().to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()  # 获取概率最大处的索引值

    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    print_res1 = "class: {}".format(class_indict[str(predict_cla)])
    print("*" * 150, '\n', print_res1)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("循环运行时间:%.2f秒" % (end - start))