import numpy as np
import torch
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class HrrpDataset(Dataset):
    def __init__(self, path):
        self.xy = pd.read_csv(path, header=None)
        xy = self.xy
        x = xy.iloc[:, :-1].values
        x = self.x_norm(x)
        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x.float(), dim=1)
        label = xy.iloc[:, -1]
        enc = LabelEncoder()  # 获取一个LabelEncoder
        labels = ['A320', 'A321', 'A330', 'A350', 'ARJ21', 'BY737', 'BY777', 'BY787']
        enc = enc.fit(labels)
        self.y = enc.transform(label)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

    @classmethod
    def x_norm(cls, x):
        _x_norm = np.linalg.norm(x, ord=None, axis=1, keepdims=False).reshape(-1, 1)
        x = x / _x_norm
        return x


"""网络测试"""
batch_size = 2728
test_path = '/home/private/hrrp/NanHu/20230419/test/test.csv'
dataset = HrrpDataset(test_path)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=os.cpu_count(), shuffle=False,
                         drop_last=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_weight_path = "../weight/20230419Resnet34/20230419Resnet34Best.pth"

assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
net = torch.load(model_weight_path, map_location=device)


net.eval()
with torch.no_grad():
    for test_data in tqdm(data_loader):
        test_images, _ = test_data
        outputs0 = net(test_images.to(device))

hrrpTest = test_images.squeeze()
# dataT = hrrpTest.cpu().numpy()
data = outputs0.cpu().numpy()
data = np.reshape(data, (8, -1))

# 计算类内距离
intra_distances = np.zeros(9)
for i in range(9):
    mean = np.mean(data[i], axis=0)
    temp = np.sum((data[i] - mean) ** 2, axis=1)
    intra_distances[i] = np.mean(temp)

# 计算类间距离
inter_distances = np.zeros((9, 9))
for i in range(8):
    for j in range(i + 1, 9):
        mean_i = np.mean(data[i], axis=0)
        mean_j = np.mean(data[j], axis=0)
        inter_distances[i][j] = np.sum((mean_i - mean_j) ** 2)
        inter_distances[j][i] = inter_distances[i][j]

# 计算类内距离和类间距离比
intra_mean = np.mean(intra_distances)
inter_mean = np.mean(inter_distances)
ratio = intra_mean / inter_mean

print("类内距离平均值：", intra_mean)
print("类间距离平均值：", inter_mean)
print("类内距离和类间距离比：", ratio)
