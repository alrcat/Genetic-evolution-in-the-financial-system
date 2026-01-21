"""配置模块"""

import os
import yaml
from typing import Any, Dict

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(config_name: str = "world_rules.yaml") -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_name: 配置文件名
        
    Returns:
        配置字典
    """
    config_path = os.path.join(CONFIG_DIR, config_name)
    
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
