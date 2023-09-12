import argparse
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts  # 需要安装的库
import os
import time
import torch
import numpy as np
import torch.optim as optim
import random
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter  # tensorboard可能要安装
from model import resnet18, resnet34, resnet50, resnet101  # 总共四个模型，哪个模型好用哪个
from utils.dataset import my_dataloader
from utils.train_eval_utils import train_one_epoch, evaluate


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    '''训练日志'''

    tb_writer = SummaryWriter(log_dir=args.logs_path)

    batch_size = args.batch_size
    train_path = args.train_path
    val_path = args.val_path
    class_indices_json_path = args.class_indices_json_path
    nw = os.cpu_count()
    print('Using {} dataloader workers every process'.format(nw))

    '''实例化训练集验证集,并加载'''
    train_dataloader, train_size = my_dataloader(train_path, class_indices_json_path, batch_size=batch_size, nw=nw)
    val_dataloader, val_size = my_dataloader(val_path, class_indices_json_path, batch_size=args.val_size, nw=nw)

    print("using {} HRRP datas for training, {} HRRP data for validation.".format(train_size,
                                                                                  val_size))
    # 如果存在预训练权重则载入
    model = resnet18(args.num_classes).to(device)   # 修改模型

    # init_hrrp = torch.zeros((2, 1, 256), device=device)
    # tb_writer.add_graph(model, init_hrrp)
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # pg = [p for p in model.parameters() if p.requires_grad]  # 保留需要学习的参数，并构成列表
    weight_decay = args.weight_decay
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=weight_decay)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay, amsgrad=True)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay, amsgrad=True)
    else:
        print("没有该优化器")
        exit()

    # TODO
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=args.T_max,
                                              cycle_mult=1.0,
                                              max_lr=args.lr,
                                              min_lr=1e-7,
                                              warmup_steps=args.warmup_steps,
                                              gamma=0.8)
    best_acc = 0.0
    epochAcc = 0
    tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_dataloader,
                                                device=device,
                                                epoch=epoch,
                                                label_smoothing=args.label_smoothing,
                                                )
        tb_writer.add_scalars('lr', {tags[4]: optimizer.param_groups[0]["lr"]}, epoch)
        scheduler.step()  # 调整学习率

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_dataloader,
                                     device=device,
                                     epoch=epoch,
                                     label_smoothing=args.label_smoothing
                                     )

        tb_writer.add_scalars('loss', {tags[0]: train_loss, tags[2]: val_loss}, epoch)
        tb_writer.add_scalars('acc', {tags[1]: train_acc, tags[3]: val_acc}, epoch)

        # for name, param in model.named_parameters():
        #     tb_writer.add_histogram(tag=name + '_data', values=param.data, global_step=epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            epochAcc = epoch
            savefile = args.save_path + 'best.pth'
            torch.save(model.state_dict(), savefile)

        # if val_acc > best_acc:
        #     best_acc = val_acc
        #     epochAcc = epoch
        #     savefile = '{}.pth'.format(epoch + 1)
        #     torch.save(model, args.save_path + savefile)

    print("Train finishing! The best_acc:{:.2f}%. EpochAcc:{}".format(
        best_acc * 100, epochAcc+1))
    print("batch-size:{},lr:{}||save_path:{}||optimizer:{}||weights:{}||T_max:{}||weight_decay:{}||logs_path:{}"
          "||save_path:{}||"
          "label_smoothing:{}".format(args.batch_size, args.lr, args.save_path, args.optimizer, args.weights,
                                      args.T_max,
                                      args.weight_decay, args.logs_path, args.save_path, args.label_smoothing))

    print('Start Tensorboard with "tensorboard --logdir={}", view at http://localhost:6006/'.format(args.logs_path))


if __name__ == '__main__':
    """
    num_classes
    epochs
    batch-size
    lr
    save_path
    optimizer
    train_path
    val_path
    weights
    freeze-layers
    device
    T_max
    """

    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--warmup_steps', type=int, default=5)
    parser.add_argument('--class_indices_json_path', type=str, default="./class_indices.json")
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=None)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--val_size', type=int, default=16810)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--T_max', type=int, default=20)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--logs_path', type=str, default="./log/9_12")
    parser.add_argument('--save_path', type=str, default='./weight/9_12/')
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--train_path', type=str,
                        default="data/9.12/traindata_013489.csv")  # 训练数据集
    parser.add_argument('--val_path', type=str,
                        default="data/9.12/testdata_013489.csv")
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    opt = parser.parse_args()

    setup_seed(42)
    main(opt)

    end = time.time()
    print("训练花费时间：{}秒".format(time.strftime("%H:%M:%S", time.gmtime(end - start))))
