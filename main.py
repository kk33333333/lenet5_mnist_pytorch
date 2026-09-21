# -*- coding: utf-8 -*-
"""
@Author : kk
@Date : 2026/9/21 14:52
@Describe : 主程序
"""
from utils import init_env
from config import DEVICE, LR, MOMENTUM, EPOCHS
from dataset import get_dataloaders
from model import LeNet5
from train_eval import train_loop, plot_curves
from torch import nn
from torch import optim

if __name__ == "__main__":
    init_env()
    print(f"使用设备: {DEVICE}")

    train_loader, test_loader = get_dataloaders()
    model = LeNet5().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    train_loss_list, test_loss_list, train_acc_list, test_acc_list = train_loop(
        model, train_loader, test_loader, loss_fn, optimizer, EPOCHS
    )
    plot_curves(train_loss_list, test_loss_list, train_acc_list, test_acc_list)

    print("训练完成！")
    print(f"最优权重 best.pt 和最终权重 last.pt 保存在 runs/weights")
    print(f"损失精度曲线图保存在 runs/figures")
