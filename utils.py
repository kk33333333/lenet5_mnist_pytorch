# -*- coding: utf-8 -*-
"""
@Author : kk
@Date : 2026/9/21 14:53
@Describe : 工具函数
"""

import os
import ssl

def init_env():
    # 解决OpenMP库冲突（OMP Error #15）
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # 解决MNIST下载ssl证书报错
    ssl._create_default_https_context = ssl._create_unverified_context
