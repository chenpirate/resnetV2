import random

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.backends import cudnn

from tSNE2D import tSNE
import os
import torch
import numpy as np
from tqdm import tqdm
from model import resnet18, resnet34, resnet50, resnet101
from utils.dataset import my_dataloader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


if __name__ == '__main__':
    setup_seed(42)
    title = f'20230501\n'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    enc = LabelEncoder()  # 获取一个LabelEncoder
    # labels = ['A330', 'A350', 'ARJ21', 'BY737', 'BY777', 'BY787']
    labels = ['A330', 'A350', 'ARJ21', 'BY737', 'BY777', 'BY787', 'A321', 'A320', 'CRJ21']
    enc = enc.fit(labels)
    print(device)
    batch_size = 21144
    test_path = '/home/private/hrrp/NanHu/resnetV2/data/20230501/test.csv'  # 测试集
    test_dataloader, test_size = my_dataloader(test_path, batch_size=batch_size, nw=16)
    print("using {} HRRP datas for testing.".format(test_size))

    net = resnet101(10).to(device)
    # # load pretrain weights
    model_weight_path = '../weight/20230501/101/best.pth'  # 权重文件
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.eval()
    with torch.no_grad():
        for test_data in tqdm(test_dataloader):
            test_images, test_labels = test_data
            outputs0 = net(test_images.to(device))
    outputs0 = outputs0.cpu().detach().numpy()
    test_labels = test_labels.cpu().detach().numpy()
    test_labels = np.squeeze(test_labels)
    test_labels = enc.inverse_transform(test_labels)

    outputs0 = pd.DataFrame(outputs0)
    labels = pd.DataFrame(test_labels)
    labels = labels.iloc[:, 0]

    tSNE = tSNE()
    tSNE.visual_dataset(outputs0, labels, title)

    # outputs1 = pd.DataFrame(outputs1)
    # title = f'output1\n'
    # tSNE.visual_dataset(outputs1, labels, title)
