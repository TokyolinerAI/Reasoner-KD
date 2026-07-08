# D:/code/model/my/student/run.py

import sys
import os

# 将项目根目录 (student) 和 src 目录都添加到 Python 路径中
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_CURRENT_DIR, 'src')

# 1. 优先添加项目根目录（确保 eval_kit 能被找到，因 eval_kit 在 student 目录下）
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)  # 新增：项目根目录加入搜索路径

# 2. 添加 src 目录（原有逻辑保留）
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


# 从 src 包中导入 runner 模块
from runner import main # 因为 src 已经在路径里了

if __name__ == '__main__':
    main()