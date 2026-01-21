"""
资源管理系统 - Agent的资源约束

核心原则:
- Agent是资源受限实体
- 资源消耗是真实的，不能凭空产生
- 资源耗尽导致死亡
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional
import uuid


class ResourceType(Enum):
    """资源类型枚举"""
    CAPITAL = auto()        # 资本：用于交易和维持生存
    RISK_BUDGET = auto()    # 风险预算：可承受的风险上限
    ATTENTION = auto()      # 注意力：能同时关注的标的数量
    COMPUTE = auto()        # 算力：决策的计算复杂度上限
    ENERGY = auto()         # 能量：执行行动的能力


@dataclass
class ResourceConfig:
    """资源配置"""
    initial_value: float
    min_value: float = 0.0
    max_value: float = float('inf')
    decay_rate: float = 0.0  # 每tick的自然衰减率
    regeneration_rate: float = 0.0  # 每tick的自然恢复率


# 默认资源配置
DEFAULT_RESOURCE_CONFIGS: Dict[ResourceType, ResourceConfig] = {
    ResourceType.CAPITAL: ResourceConfig(
        initial_value=10000.0,
        min_value=0.0,
        max_value=float('inf'),
        decay_rate=0.0,  # 资本不自然衰减
    ),
    ResourceType.RISK_BUDGET: ResourceConfig(
        initial_value=1000.0,
        min_value=0.0,
        max_value=1000.0,
        decay_rate=0.0,
        regeneration_rate=10.0,  # 风险预算会缓慢恢复
    ),
    ResourceType.ATTENTION: ResourceConfig(
        initial_value=5.0,  # 能同时关注5个标的
        min_value=1.0,
        max_value=20.0,
        decay_rate=0.0,
    ),
    ResourceType.COMPUTE: ResourceConfig(
        initial_value=100.0,
        min_value=0.0,
        max_value=100.0,
        regeneration_rate=100.0,  # 每tick完全恢复
    ),
    ResourceType.ENERGY: ResourceConfig(
        initial_value=100.0,
        min_value=0.0,
        max_value=100.0,
        decay_rate=1.0,  # 能量会自然消耗
        regeneration_rate=5.0,  # 但也会恢复
    ),
}


@dataclass
class AgentResources:
    """
    Agent的资源账户
    
    这是Agent的生存基础。资源的消耗和获取决定了Agent的生死。
    
    关键约束:
    - 资本归零 = 饿死
    - 风险预算归零 = 强制平仓
    - 能量归零 = 无法行动
    """
    
    resource_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # 核心资源
    capital: float = 10000.0          # 可用资本
    risk_budget: float = 1000.0       # 风险预算
    attention_units: int = 5          # 注意力单位
    compute_quota: float = 100.0      # 计算配额
    energy: float = 100.0             # 能量
    
    # 代谢率（维持生存的成本）
    metabolism_rate: float = 1.0      # 每tick消耗的资本
    
    # 配置
    configs: Dict[ResourceType, ResourceConfig] = field(
        default_factory=lambda: DEFAULT_RESOURCE_CONFIGS.copy()
    )
    
    def consume_capital(self, amount: float) -> bool:
        """
        消耗资本
        
        Args:
            amount: 消耗数量
            
        Returns:
            是否成功消耗（资本不足时返回False）
        """
        if amount < 0:
            raise ValueError("消耗数量不能为负")
        
        if self.capital >= amount:
            self.capital -= amount
            return True
        return False
    
    def add_capital(self, amount: float) -> None:
        """
        增加资本（交易盈利等）
        
        Args:
            amount: 增加数量
        """
        if amount < 0:
            raise ValueError("增加数量不能为负，请使用consume_capital")
        self.capital += amount
    
    def consume_risk_budget(self, amount: float) -> bool:
        """
        消耗风险预算
        
        Args:
            amount: 风险暴露量
            
        Returns:
            是否有足够的风险预算
        """
        if self.risk_budget >= amount:
            self.risk_budget -= amount
            return True
        return False
    
    def restore_risk_budget(self, amount: float) -> None:
        """恢复风险预算（平仓后）"""
        config = self.configs[ResourceType.RISK_BUDGET]
        self.risk_budget = min(self.risk_budget + amount, config.max_value)
    
    def consume_energy(self, amount: float) -> bool:
        """
        消耗能量
        
        Args:
            amount: 能量消耗
            
        Returns:
            是否有足够能量
        """
        if self.energy >= amount:
            self.energy -= amount
            return True
        return False
    
    def consume_compute(self, amount: float) -> bool:
        """
        消耗计算配额
        
        Args:
            amount: 计算消耗
            
        Returns:
            是否有足够配额
        """
        if self.compute_quota >= amount:
            self.compute_quota -= amount
            return True
        return False
    
    def pay_metabolism(self) -> bool:
        """
        支付代谢成本（每tick调用）
        
        Returns:
            是否能支付（资本不足时返回False，意味着濒临死亡）
        """
        return self.consume_capital(self.metabolism_rate)
    
    def regenerate(self) -> None:
        """
        资源再生（每tick调用）
        
        某些资源会自然恢复
        """
        # 风险预算恢复
        risk_config = self.configs[ResourceType.RISK_BUDGET]
        self.risk_budget = min(
            self.risk_budget + risk_config.regeneration_rate,
            risk_config.max_value
        )
        
        # 计算配额恢复
        compute_config = self.configs[ResourceType.COMPUTE]
        self.compute_quota = min(
            self.compute_quota + compute_config.regeneration_rate,
            compute_config.max_value
        )
        
        # 能量恢复（但也有自然衰减）
        energy_config = self.configs[ResourceType.ENERGY]
        self.energy = max(0, self.energy - energy_config.decay_rate)
        self.energy = min(
            self.energy + energy_config.regeneration_rate,
            energy_config.max_value
        )
    
    def is_bankrupt(self) -> bool:
        """检查是否破产（资本归零）"""
        return self.capital <= 0
    
    def is_risk_exhausted(self) -> bool:
        """检查风险预算是否耗尽"""
        return self.risk_budget <= 0
    
    def is_exhausted(self) -> bool:
        """检查能量是否耗尽"""
        return self.energy <= 0
    
    def get_survival_status(self) -> Dict[str, bool]:
        """获取生存状态摘要"""
        return {
            "can_survive": not self.is_bankrupt(),
            "can_trade": not self.is_risk_exhausted() and not self.is_exhausted(),
            "has_capital": self.capital > 0,
            "has_risk_budget": self.risk_budget > 0,
            "has_energy": self.energy > 0,
        }
    
    def clone(self, inheritance_ratio: float = 0.5) -> AgentResources:
        """
        克隆资源（用于繁殖）
        
        Args:
            inheritance_ratio: 继承比例（后代获得的资源比例）
            
        Returns:
            新的资源账户
        """
        return AgentResources(
            capital=self.capital * inheritance_ratio,
            risk_budget=self.configs[ResourceType.RISK_BUDGET].initial_value,
            attention_units=self.attention_units,
            compute_quota=self.configs[ResourceType.COMPUTE].initial_value,
            energy=self.configs[ResourceType.ENERGY].initial_value,
            metabolism_rate=self.metabolism_rate,
            configs=self.configs.copy(),
        )
    
    def __repr__(self) -> str:
        return (
            f"AgentResources(capital={self.capital:.2f}, "
            f"risk={self.risk_budget:.2f}, "
            f"energy={self.energy:.2f})"
        )
