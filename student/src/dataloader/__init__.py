# D:/code/model/my/student/src/dataloader/__init__.py

"""
This __init__.py file makes the dataloader directory a Python package 
and exposes key classes and functions from its submodules.
"""

# 从 var_dataset.py 子模块中导入需要在包顶层访问的对象
from .var_dataset import VARDataset, sentences_collate, prepare_batch_inputs,cal_performance

# 定义 __all__，指定 `from dataloader import *` 的行为
__all__ = [
    'VARDataset',
    'sentences_collate',
    'prepare_batch_inputs',
    'cal_performance'
]