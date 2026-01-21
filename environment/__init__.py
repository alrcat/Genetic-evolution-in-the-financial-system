"""
市场环境模块

提供不可控的外部市场环境，包括市场状态、订单执行等功能。
"""

from environment.environment import (
    MarketRegime,
    MarketState,
    EnvironmentState,
    Order,
    ExecutionResult,
    Environment,
)

__all__ = [
    "MarketRegime",
    "MarketState",
    "EnvironmentState",
    "Order",
    "ExecutionResult",
    "Environment",
]