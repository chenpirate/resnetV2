import numpy as np
import pandas as pd
import glob
import re
from sklearn.model_selection import train_test_split


# 合并csv文件
def main():
    csv_list = glob.glob('/home/private/hrrp/NanHu/resnetV2/data/j机/train/*.csv')
    print('共发现%s个CSV文件' % len(csv_list))
    print('正在处理............')
    for i in csv_list:
        fr = open(i, 'r', encoding='gbk').read()
        with open('/home/private/hrrp/NanHu/resnetV2/data/j机/train.csv', 'a', encoding='gbk') as f:
            f.write(fr)
    print('合并完毕！')


if __name__ == '__main__':
    main()
