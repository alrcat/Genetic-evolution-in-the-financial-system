"""
核心模块 - 定义系统的基础组件

包含:
- WorldClock: 不可逆的世界时钟
- Resources: Agent资源管理
- Gene: 基因（最小可遗传单元）
- DNA: 基因组合体
- Agent: 资源受限的生命体
- Environment: 不可控的外部市场环境
"""

from core.world_clock import WorldClock
from resources import AgentResources, ResourceType
from core.gene import Gene, GeneType, GeneExpression
from core.dna import DNA, Chromosome, Phenotype
from core.agent import Agent, AgentState
from environment import Environment, MarketState, MarketRegime

__all__ = [
    "WorldClock",
    "AgentResources",
    "ResourceType",
    "Gene",
    "GeneType",
    "GeneExpression",
    "DNA",
    "Chromosome",
    "Phenotype",
    "Agent",
    "AgentState",
    "Environment",
    "MarketState",
    "MarketRegime",
]
