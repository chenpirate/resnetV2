import os
import json

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from model import resnet18, resnet34, resnet50, resnet101
from utils.dataset import my_dataloader


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self, accP=None):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        if accP is not None:
            if (acc*100) < accP:
                print("the model accuracy is {:.2f}%".format(acc * 100))
        else:
            print("the model accuracy is {:.2f}%".format(acc * 100))

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        colMatrix = np.sum(matrix, 0)
        matrixPercent = matrix / colMatrix
        # print(matrix)
        plt.figure(0)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')
        # 在图中标注数量/概率信息 format(info)+'%'
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        # plt.savefig('/home/private/paperLabNotebook/image/tSNE/98.61%65.svg', format='svg', bbox_inches='tight')
        plt.tight_layout()

        plt.figure(1)
        plt.imshow(matrixPercent, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')
        # 在图中标注数量/概率信息
        # for x in range(self.num_classes):
        #     for y in range(self.num_classes):
        #         matrixPercent = round(matrixPercent[y, x] * 100, 2)

        thresh = 50
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = round(matrixPercent[y, x] * 100, 2)
                plt.text(x, y, '{}%'.format(info),
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")

        # plt.savefig('/home/private/paperLabNotebook/image/tSNE/{98.61%65per}.svg', format='svg',
        #             bbox_inches='tight')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = 7634
    # 单个
    test_path = '/home/private/hrrp/NanHu/resnetV2/data/j机/test.csv'
    test_dataloader, test_size = my_dataloader(test_path, shuffle=False, batch_size=batch_size, nw=os.cpu_count())
    print("using {} HRRP datas for testing.".format(test_size))
    net = resnet18(5).to(device)

    # # load pretrain weights
    model_weight_path = '/home/private/hrrp/NanHu/resnetV2/weight/j机/best.pth'
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))

    # read class_indict
    json_label_path = '../class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=4, labels=labels)
    net.eval()
    with torch.no_grad():
        for test_data in tqdm(test_dataloader):
            test_images, test_labels = test_data
            outputs = net(test_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), test_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()


