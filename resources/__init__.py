"""
资源管理系统模块

提供Agent的资源约束和资源管理功能。
"""

from resources.resources import (
    ResourceType,
    ResourceConfig,
    DEFAULT_RESOURCE_CONFIGS,
    AgentResources,
)

__all__ = [
    "ResourceType",
    "ResourceConfig",
    "DEFAULT_RESOURCE_CONFIGS",
    "AgentResources",
]