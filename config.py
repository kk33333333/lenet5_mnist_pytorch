# -*- coding: utf-8 -*-
"""
@Author : kk
@Date : 2026/9/21 14:51
@Describe : 超参配置文件
"""
import torch

# 训练超参
BATCH_SIZE = 64
LR = 0.01
MOMENTUM = 0.9
EPOCHS = 10

# 路径配置
DATA_ROOT = "data"
WEIGHT_SAVE_DIR = "runs/weights"
FIGURE_SAVE_DIR = "runs/figures"

# 设备自动选择
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像预处理尺寸
IMG_SIZE = (32, 32)
