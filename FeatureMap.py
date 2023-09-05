import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch import Tensor

from resnet34 import resnet34
import torch.nn as nn
from torchinfo import summary
import torch

from typing import Dict, Iterable, Callable


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output

        return fn

    # : 函数参数中的冒号是参数的类型建议符，此处建议输入实参为Tensor类型。-> 函数后面跟着的箭头是函数返回值的类型建议符，此处建议函数返回值类型为字典，键值类型分别str，Tensor。
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features


def x_norm(x):
    x_norm = np.linalg.norm(x, ord=None, axis=1, keepdims=False).reshape(-1, 1)
    x = x / x_norm
    return x


if __name__ == "__main__":
    test_path = '/home/pirate/HRRP/algorithm_transformer/data/完备方位角hrrp/test/56°test.csv'
    xy = pd.read_csv(test_path, header=None)
    x = xy.iloc[:, :-1].values
    x = x_norm(x)
    input0 = torch.from_numpy(x)
    input0 = torch.unsqueeze(input0.float(), dim=1)
    net = resnet34(12)
    model_weight_path = "../resnet/weight/20221118.pth"
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path))
    print(net)
    resnet_features = FeatureExtractor(net, layers=["layer1.2.bn2"])
    features = resnet_features(input0)
    conv4 = features['layer1.2.bn2'].detach().numpy()

    conv4 = np.mean(conv4, keepdims=True, axis=1)
    conv4 = np.squeeze(conv4)
    conv4 = np.transpose(conv4, [1, 0])
    conv4 = np.matrix(conv4)
    ax1 = plt.matshow(conv4, aspect='auto', interpolation='bilinear', cmap=plt.cm.Reds)
    plt.colorbar(ax1.colorbar, fraction=0.025)
    ax2 = plt.matshow(x, aspect='auto', interpolation='bilinear', cmap=plt.cm.Reds)
    plt.colorbar(ax2.colorbar, fraction=0.025)
    plt.show()


