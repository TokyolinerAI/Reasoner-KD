#!/usr/bin/env python
"""
数据完整性检查脚本
使用方法: python check_data.py
"""

import os
import json
import numpy as np


def check_var_data():
    """检查VAR数据集的完整性"""
    data_dir = "D:/code/model/VAR-main/data/VAR"

    print("=" * 60)
    print("检查VAR数据集完整性")
    print("=" * 60)

    # 1. 检查目录结构
    required_dirs = [
        "data",
        "video_feature",
        "vocab_feature"
    ]

    for dir_name in required_dirs:
        dir_path = os.path.join(data_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"✓ {dir_name}/ 目录存在")
        else:
            print(f"✗ {dir_name}/ 目录不存在！")
            return False

    # 2. 检查数据文件
    data_files = [
        "data/var_train_v1.0.json",
        "data/var_val_v1.0.json",
        "data/var_test_v1.0.json"
    ]

    for file_name in data_files:
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                print(f"✓ {file_name}: {len(data)} 个样本")
        else:
            print(f"✗ {file_name} 不存在！")
            return False

    # 3. 检查词汇特征文件
    vocab_files = [
        "vocab_feature/min3_vocab_glove_6B.pt",
        "vocab_feature/min3_word2idx_6B.json"
    ]

    for file_name in vocab_files:
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            print(f"✓ {file_name} 存在")
        else:
            print(f"✗ {file_name} 不存在！")
            print(f"   请从 https://pan.baidu.com/s/1Ju6O-05IhdVsNvpgbVpD7g 下载")
            return False

    # 4. 检查视频特征文件
    video_feat_dir = os.path.join(data_dir, "video_feature")
    if os.path.exists(video_feat_dir):
        feat_files = os.listdir(video_feat_dir)
        print(f"✓ video_feature/: {len(feat_files)} 个特征文件")
        if len(feat_files) == 0:
            print(f"✗ video_feature/ 目录为空！")
            print(f"   请从 https://pan.baidu.com/s/1Ju6O-05IhdVsNvpgbVpD7g 下载")
            return False

    # 5. 检查测试集大小（应该是1093个样本）
    test_file = os.path.join(data_dir, "data/var_test_v1.0.json")
    with open(test_file, 'r') as f:
        test_data = json.load(f)
        if len(test_data) == 1093:
            print(f"✓ 测试集大小正确: {len(test_data)} 个样本")
        else:
            print(f"✗ 测试集大小不正确: {len(test_data)} 个样本（期望1093个）")
            return False

    print("\n" + "=" * 60)
    print("数据检查完成！所有文件都存在且格式正确。")
    print("=" * 60)
    return True


def check_model_dir():
    """检查模型保存目录"""
    model_dir = "/d/code/model/VAR-main/model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"✓ 创建模型保存目录: {model_dir}")
    else:
        print(f"✓ 模型保存目录已存在: {model_dir}")


if __name__ == "__main__":
    try:
        success = check_var_data()
        check_model_dir()

        if success:
            print("\n🎉 数据检查通过！可以开始训练。")
            print("\n运行训练命令:")
            print("bash scripts/train.sh 1 Reasoner")
        else:
            print("\n❌ 数据检查失败！请先完成数据准备。")

    except Exception as e:
        print(f"检查过程中出现错误: {e}")