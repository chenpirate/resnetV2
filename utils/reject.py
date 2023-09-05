import os
import json

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from model import resnet18, resnet34, resnet50, resnet101
from utils.dataset import my_dataloader


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = 1
    test_path = '/home/private/hrrp/NanHu/20230419/test_reject.csv'
    test_dataloader, test_size = my_dataloader(test_path, batch_size=batch_size, nw=16)
    print("using {} HRRP datas for testing.".format(test_size))

    # load pretrain weights
    model_weight_path = "/home/private/hrrp/NanHu/resnetV2/weight/20230419Resnet34/19.pth"
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    net = torch.load(model_weight_path, map_location=device)
    # net.to(device)

    # read class_indict
    json_label_path = '../class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    count = 0
    net.eval()
    with torch.no_grad():
        for test_data in tqdm(test_dataloader):
            test_images, test_labels = test_data
            outputs = net(test_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            if outputs.max() > 0.85:
                count += 1
                print('\n', outputs.max())
                outputs = torch.argmax(outputs, dim=1)
                print(outputs)
        print(count, "/", test_size)


if __name__ == '__main__':
    main()