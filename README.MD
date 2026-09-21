LeNet-5 手写数字识别实验（PyTorch）

## 项目简介
本项目基于 PyTorch 框架实现经典的 LeNet-5 卷积神经网络，完成 MNIST 手写数字数据集的 10 分类任务。
项目采用模块化设计，将网络划分为：
- BackBone：负责卷积特征提取
- Head：负责特征映射与分类
整体网络结构清晰，训练、评估、可视化以及模型权重保存等功能分别进行组织。

## 项目结构
LeNet_5/
├── config.py              # 超参数与路径统一配置
├── utils.py               # 环境初始化工具
├── dataset.py             # MNIST 数据集加载与预处理
├── model.py               # 网络结构定义（BackBone / Head / LeNet5）
├── train_eval.py          # 训练、评估与结果可视化
├── main.py                # 程序主入口
├── .gitignore             # Git 文件过滤规则
├── data/                  # MNIST 数据集
└── runs/
    ├── figures/           # 损失与准确率曲线
    └── weights/           # 模型权重
        └── best.pt        # 测试集表现最优的模型权重

## 环境依赖
- Python 3.8 及以上版本
- PyTorch
- TorchVision
- Matplotlib


## 模型结构
LeNet-5 网络主要由 BackBone（卷积特征提取） 和 Head（全连接分类） 两部分组成。
### BackBone（卷积特征提取模块）

负责图像特征提取，结构如下：

输入图像
   ↓
Conv1
1 → 6，5×5，Stride=1，Padding=0
   ↓
Tanh
   ↓
AvgPool1
2×2，Stride=2
   ↓
Conv2
6 → 16，5×5，Stride=1，Padding=0
   ↓
Tanh
   ↓
AvgPool2
2×2，Stride=2

- 激活函数：Tanh

### Head（全连接分类模块）

负责特征映射到分类结果，结构如下：

16 × 5 × 5
    ↓
Flatten
    ↓
400
    ↓
Linear
400 → 120
    ↓
Tanh
    ↓
Linear
120 → 84
    ↓
Tanh
    ↓
Linear
84 → 10

- 前两层激活函数：Tanh

## 训练流程
模型训练的基本流程为：
读取 MNIST 数据
      ↓
数据预处理
      ↓
输入模型
      ↓
前向传播
      ↓
计算 Loss
      ↓
反向传播
      ↓
计算梯度
      ↓
更新模型参数
      ↓
重复训练

## 实验结果

本次实验在 CPU 环境下训练 10 轮，结果如下：

- 训练集最终准确率：**99.30%**
- 测试集最高准确率：**98.84%**
- 损失曲线平稳下降，准确率曲线稳步上升，模型收敛良好，无明显过拟合现象。
- 训练过程中自动保存测试集表现最优的模型权重`best.pt`，以及训练结束时的最终权重`shturl.`。
- 损失与准确率随 Epoch 变化的可视化曲线保存在`runs/figures/loss_acc_curve.png`。

## 总结
本项目通过 PyTorch 实现经典 LeNet-5 网络，并将网络划分为 BackBone 和 Head 两个主要模块。
其中：
- BackBone 负责从图像中提取特征；
- Head 负责将提取到的特征映射到 10 个数字类别；
- 通过前向传播、Loss 计算、反向传播和参数更新，使模型逐渐学习 MNIST 手写数字的特征。
最终在 CPU 环境下训练 10 轮，训练集最终准确率达到 99.30%，测试集最高准确率达到 98.84%。