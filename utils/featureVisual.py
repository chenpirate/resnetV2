from swin_transformer_1d import swin_transformer
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl


def x_norm(x):
    x_norm1 = np.linalg.norm(x, ord=None, axis=1, keepdims=False).reshape(-1, 1)
    x = x / x_norm1
    return x


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
net = swin_transformer(img_size=256, num_classes=12, in_chans=1, patch_size=4,
                       depths=[2, 2, 6, 2]).to(device)
model_weight_path = "/home/private/paperWight/finalv6/20230211.pth"
net.load_state_dict(torch.load(model_weight_path, map_location=device))

for name, _ in net.named_modules():
    print(name)

fmap_block = list()
input_block = list()


def hook(module, inputdata, output):
    fmap_block.append(output)  # 记录feature map
    input_block.append(inputdata)


net.layers[0].blocks[0].proj.register_forward_hook(hook)
net.layers[0].blocks[1].proj.register_forward_hook(hook)
net.layers[1].blocks[0].proj.register_forward_hook(hook)
net.layers[1].blocks[1].proj.register_forward_hook(hook)

path = '/home/private/hrrp/data/test/56.csv'

xy = pd.read_csv(path, header=None)
data = x_norm(xy.iloc[:, :-1].values)
HRRP = torch.unsqueeze(torch.from_numpy(data).float(), 1)
img = HRRP[:1]
xa = net(img)

fT1 = input_block[1][0][0].permute(1, 0).detach().numpy()
fT2 = fmap_block[1][0].permute(1, 0).detach().numpy()
fT3 = input_block[2][0][0].permute(1, 0).detach().numpy()
fT4 = fmap_block[3][0].permute(1, 0).detach().numpy()

plt.figure(1)
plt.subplot(221)
plt.imshow(fT1, norm=mpl.colors.Normalize(0, 1), aspect='auto', interpolation='gaussian', cmap=plt.cm.hot)
plt.colorbar()
plt.subplot(222)
plt.imshow(fT2, norm=mpl.colors.Normalize(0, 1), aspect='auto', interpolation='gaussian', cmap=plt.cm.hot)
plt.colorbar()
plt.subplot(223)
plt.imshow(fT3, norm=mpl.colors.Normalize(0, 1), aspect='auto', interpolation='gaussian', cmap=plt.cm.hot)
plt.colorbar()
plt.subplot(224)
plt.imshow(fT4, norm=mpl.colors.Normalize(0, 1), aspect='auto', interpolation='gaussian', cmap=plt.cm.hot)
plt.colorbar()
plt.tight_layout()

# plt.figure(2)
# plt.subplot(121)
# plt.plot(fT1[43])
# plt.subplot(122)
# plt.plot(fT1[43])
# plt.tight_layout()
# # 把画布分为三行五列，并设置figure标题
# fig, axs0 = plt.subplots(3, 4, figsize=(15, 15))
# fig.suptitle('use .suptitle() to add a figure title')

# # 用for循环在第一二行的所有子图框中作图
# cnt = 0
# for i in range(3):
#     for j in range(4):
#         axs0[i][j].imshow(fmap_block[0][0][cnt].cpu().detach().numpy())
#         cnt += 1
#         if cnt == 11:
#             cnt = 0



plt.show()
