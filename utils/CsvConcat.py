import pandas as pd


def aug2(path, save):
    df1 = pd.read_csv(path[0], header=None)
    df2 = pd.read_csv(path[1], header=None)
    data = pd.concat([df1, df2], axis=0)
    data.to_csv(save, index_label=False, header=False, index=False)
    print("f")


def aug3(path, save):
    df0 = pd.read_csv(path[0], header=None)
    df2 = pd.read_csv(path[1], header=None)
    df3 = pd.read_csv(path[2], header=None)
    data = pd.concat([df0, df2, df3], axis=0)
    data.to_csv(save, index_label=False, header=False, index=False)
    print("f")


def aug4(path, save):
    df0 = pd.read_csv(path[0], header=None)
    df1 = pd.read_csv(path[1], header=None)
    df2 = pd.read_csv(path[2], header=None)
    df3 = pd.read_csv(path[3], header=None)
    data = pd.concat([df0, df1, df2, df3], axis=0)
    data.to_csv(save, index_label=False, header=False, index=False)
    print("f")


def aug5(path, save):
    df0 = pd.read_csv(path[0], header=None)
    df1 = pd.read_csv(path[1], header=None)
    df2 = pd.read_csv(path[2], header=None)
    df3 = pd.read_csv(path[3], header=None)
    df4 = pd.read_csv(path[4], header=None)
    data = pd.concat([df0, df1, df2, df3, df4], axis=0)
    data.to_csv(save, index_label=False, header=False, index=False)
    print("f")


def aug6(path, save):
    df0 = pd.read_csv(path[0], header=None)
    df1 = pd.read_csv(path[1], header=None)
    df2 = pd.read_csv(path[2], header=None)
    df3 = pd.read_csv(path[3], header=None)
    df4 = pd.read_csv(path[4], header=None)
    df5 = pd.read_csv(path[5], header=None)
    data = pd.concat([df0, df1, df2, df3, df4, df5], axis=0)
    data.to_csv(save, index_label=False, header=False, index=False)
    print("f")


def aug7(path, save):
    df0 = pd.read_csv(path[0], header=None)
    df1 = pd.read_csv(path[1], header=None)
    df2 = pd.read_csv(path[2], header=None)
    df3 = pd.read_csv(path[3], header=None)
    df4 = pd.read_csv(path[4], header=None)
    df5 = pd.read_csv(path[5], header=None)
    df6 = pd.read_csv(path[6], header=None)
    data = pd.concat([df0, df1, df2, df3, df4, df5, df6], axis=0)
    data.to_csv(save, index_label=False, header=False, index=False)
    print("f")


if __name__ == '__main__':


    noisePath7 = ['/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/6°/aug/5db.csv',
                  '/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/6°/aug/10db.csv',
                  '/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/6°/aug/15db.csv',
                  '/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/6°/aug/20db.csv',
                  '/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/6°/aug/25db.csv',
                  '/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/6°/aug/30db.csv',
                  '/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/6°/aug/35db.csv',
                  ]
    # save = '/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/6°/aug/noise.csv'
    # aug7(noisePath7, save)

    # BoxCoxPath6 = ['/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/6°/aug/0.5Box.csv',
    #                '/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/6°/aug/0.7Box.csv',
    #                '/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/6°/aug/1.2Box.csv',
    #                '/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/6°/aug/1.4Box.csv',
    #                '/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/6°/aug/AdaptiveBox.csv',
    #                '/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/6°/aug/lnBox.csv',
    #                ]
    # save = '/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/6°/aug/Box.csv'
    # aug6(BoxCoxPath6, save)

    # BoxFlipNoiseRollTrainPath5 = [
    #     '/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/6°/aug/Box.csv',
    #     '/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/6°/aug/flip.csv',
    #     '/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/6°/aug/noise.csv',
    #     '/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/6°/aug/roll.csv',
    #     '/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/6°/aug/train.csv'
    # ]
    # save = '/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/6°/aug/BoxFlipNoiseRollTrain.csv'
    # aug5(BoxFlipNoiseRollTrainPath5, save)

    # noiseTrainPath2 = ['/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/3°/aug/train.csv',
    #               '/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/3°/aug/noise.csv', ]
    # BoxTrainPath2 = ['/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/3°/aug/train.csv',
    #             '/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/3°/aug/Box.csv']

    # save = '/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/train/3°/aug/BoxTrain.csv'






