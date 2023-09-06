import sys

import torch.nn as nn
from tqdm import tqdm
import torch
# from utils.centerloss import CenterLoss
from utils.map_tensor_to_dist import map_tensor_to_dist


def train_one_epoch_reject(model, optimizer, data_loader, device, epoch,
                    label_smoothing):
    """
    一个epoch里要做的操作
    Args:
    :param label_smothing: 标签平滑
    :param model: 模型
    :param optimizer:优化器
    :param data_loader: 训练数据加载器
    :param device: cpu 还是gpu
    :param epoch: 第几个epoch
    :return:这个epoch训练损失，训练精度


        lam: 0
        label_smoothing: 标签平滑
    """
    model.train()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()
    CEL = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    KL_loss = nn.KLDivLoss(reduction='batchmean')
    sample_num = 0

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        HRRP, labels = data
        sample_num += HRRP.shape[0]

        pred, d_reduction = model(HRRP.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.long().to(device)).sum()

        pred_dist = map_tensor_to_dist(pred, dist_type="gaussian")
        d_reduction_dist = map_tensor_to_dist(d_reduction, dist_type="t")
        
        kl_loss = KL_loss(pred_dist.log(), d_reduction_dist)
        CEL_lose = CEL(pred, labels.long().to(device))
        loss = CEL_lose + 10*kl_loss
        loss.backward()

        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] train loss: {:.3f}, acc: {:.2f}%".format(epoch + 1,
                                                                                      accu_loss.item() / (step + 1),
                                                                                      (accu_num.item() / sample_num)
                                                                                      * 100)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate_reject(model, data_loader, device, epoch, label_smoothing):
    """
    验证评估函数
    :param model: 模型
    :param data_loader: 验证加载器
    :param device: cpu 还是gpu
    :param epoch: 第几个epoch
    :return: 这个epoch验证损失，验证精度

    Args:
        label_smoothing: 标签平滑
        lam: center loss: 的权值
    """
    model.eval()
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    CEL = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    KL_loss = nn.KLDivLoss(reduction='batchmean')

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    with torch.no_grad():
        for step, data in enumerate(data_loader):
            hrrp, labels = data
            sample_num += hrrp.shape[0]

            pred, d_reduction = model(hrrp.to(device))
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.long().to(device)).sum()

            # loss = loss_function(pred, labels.long().to(device))
            pred_dist = map_tensor_to_dist(pred, dist_type="gaussian")
            d_reduction_dist = map_tensor_to_dist(d_reduction, dist_type="t")
            
            kl_loss = KL_loss(pred_dist.log(), d_reduction_dist)
            CEL_lose = CEL(pred, labels.long().to(device))
            loss = CEL_lose + 10*kl_loss
            accu_loss += loss

            data_loader.desc = "[valid epoch {}] valid loss: {:.3f}, acc: {:.2f}%".format(epoch + 1,
                                                                                          accu_loss.item() / (step + 1),
                                                                                          (accu_num.item() / sample_num)
                                                                                          * 100)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
