#!/usr/bin/env python3
"""
快速启动脚本

使用方法:
    python run.py              # 运行默认1000 ticks
    python run.py --ticks 5000 # 运行5000 ticks
    python run.py --help       # 查看帮助
"""

import sys
import os

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.demo import main

if __name__ == "__main__":
    main()
