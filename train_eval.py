# -*- coding: utf-8 -*-
"""
@Author : kk
@Date : 2026/9/21 14:52
@Describe : 训练、评估、绘图函数
"""
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from config import DEVICE, WEIGHT_SAVE_DIR, FIGURE_SAVE_DIR

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def evaluate(model, test_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    correct_num = 0
    total_num = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            output = model(images)
            loss = loss_fn(output, labels)
            total_loss += loss.item() * labels.size(0)
            pred = torch.argmax(output, dim=1)
            correct_num += (pred == labels).sum().item()
            total_num += labels.size(0)
    test_loss = total_loss / total_num
    test_acc = correct_num / total_num
    return test_loss, test_acc


def train_loop(model, train_loader, test_loader, loss_fn, optimizer, n_epochs):
    os.makedirs(WEIGHT_SAVE_DIR, exist_ok=True)
    train_loss_list, test_loss_list = [], []
    train_acc_list, test_acc_list = [], []
    best_acc = 0.0

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        correct_num = 0
        total_num = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(images)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            pred = torch.argmax(out, dim=1)
            correct_num += (pred == labels).sum().item()
            total_num += labels.size(0)

        train_loss = total_loss / total_num
        train_acc = correct_num / total_num
        test_loss, test_acc = evaluate(model, test_loader, loss_fn)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(f"Epoch [{epoch+1}/{n_epochs}], "
              f"train_loss: {train_loss:.4f}, train_acc: {train_acc*100:.2f}%, "
              f"test_loss: {test_loss:.4f}, test_acc: {test_acc*100:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(WEIGHT_SAVE_DIR, "best.pt"))
    torch.save(model.state_dict(), os.path.join(WEIGHT_SAVE_DIR, "last.pt"))
    return train_loss_list, test_loss_list, train_acc_list, test_acc_list


def plot_curves(train_loss_list, test_loss_list, train_acc_list, test_acc_list):
    os.makedirs(FIGURE_SAVE_DIR, exist_ok=True)
    epochs = range(1, len(train_loss_list)+1)
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, train_loss_list, label="train loss", linewidth=2)
    plt.plot(epochs, test_loss_list, label="test loss", linewidth=2)
    plt.xlabel("Epoch",fontsize=11)
    plt.ylabel("Loss",fontsize=11)
    plt.title("Loss Curve",fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1,2,2)
    plt.plot(epochs, train_acc_list, label="train acc", linewidth=2)
    plt.plot(epochs, test_acc_list, label="test acc", linewidth=2)
    plt.xlabel("Epoch",fontsize=11)
    plt.ylabel("Accuracy",fontsize=11)
    plt.title("Accuracy Curve",fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(FIGURE_SAVE_DIR, "loss_acc_curve.png")
    plt.savefig(save_path, dpi=150)
    # plt.show()
    print(f"曲线图已保存至: {save_path}")
