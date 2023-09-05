from sklearn.model_selection import train_test_split

import pandas as pd
import os


def split(path=''):
    filename = os.listdir(path)
    filenames = []
    trainpath = os.path.join(path, 'train')
    valpath = os.path.join(path, 'val')
    testpath = os.path.join(path, 'test')
    for dataname in filename:
        i = 0
        if os.path.splitext(dataname)[1] == '.csv':  # 目录下包含.json的文件
            filenames.append(dataname)
            i += 1

    name = ['train_', 'val_', 'test_']
    for cla in filenames:
        datapath = os.path.join(path, cla)
        aa = [os.path.join(trainpath, name[0] + cla), os.path.join(valpath, name[1] + cla),
              os.path.join(testpath, name[2] + cla)]
        # trainpath_save = aa[0]
        valpath_save = aa[1]
        testpath_save = aa[2]

        # aa = [os.path.join(datapath, name[0] + cla), os.path.join(datapath, name[1] + cla),
        #       os.path.join(datapath, name[2] + cla)]
        # trainpath = aa[0]
        # print(trainpath)
        # valpath = aa[1]
        # testpath = aa[2]

        df = pd.read_csv(datapath, header=None)
        label = df.iloc[:, -1]
        data = df.iloc[:, :-1]

        # x_train, x, y_train, y = train_test_split(data, label, test_size=0.50, random_state=42)
        x_test, x_val, y_test, y_val = train_test_split(data, label, test_size=0.50, random_state=42)
        # train = pd.concat([x_train, y_train], axis=1)
        val = pd.concat([x_val, y_val], axis=1)
        test = pd.concat([x_test, y_test], axis=1)

        # train.to_csv(trainpath_save, index=False, header=False)
        val.to_csv(valpath_save, index=False, header=False)
        test.to_csv(testpath_save, index=False, header=False)

    # datapath = os.path.join(path, dataname)
    # trainpath = os.path.join(path, 'train.csv')
    # valpath = os.path.join(path, 'val.csv.csv')
    # testpath = os.path.join(path, 'test.csv')
    #
    #
    #
    # train.to_csv(trainpath, index=False, header=False)
    # val.csv.to_csv(valpath, index=False, header=False)
    # test.to_csv(testpath, index=False, header=False)


if __name__ == '__main__':
    path = '/home/private/hrrp/NanHu/resnetV2/data/20230501/test/all'
    split(path)
