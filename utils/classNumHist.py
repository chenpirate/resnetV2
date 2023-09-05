import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


path = '/home/private/hrrp/NanHu/resnetV2/data/20230501/test.csv'
xy = pd.read_csv(path, header=None)
label = list(xy.iloc[:, -1])

categories = list(set(label))  # 去重后的水果种类
counts = [label.count(c) for c in categories]  # 每种水果的数量
plt.bar(categories, counts)  # 绘制柱状图
plt.xticks(range(len(categories)), categories, rotation=45)  # 设置x轴刻度和标签
for x, y in zip(categories, counts):  # 遍历每个bar的位置和数值
    plt.text(x, y + 0.1, str(y), ha="center", va="bottom")  # 在bar上方添加数值标签，水平居中，垂直底部对齐
plt.rcParams['axes.unicode_minus'] = False
plt.title('训练数据', family="AR PL UKai CN", fontsize=18)
plt.xlabel('机型', family="AR PL UKai CN", fontsize=18)
plt.ylabel('HRRP帧数', family="AR PL UKai CN", fontsize=18)
plt.show()  # 显示图形
