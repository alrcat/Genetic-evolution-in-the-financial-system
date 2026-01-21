"""
基因系统 - 最小可遗传单元

核心原则:
- 基因不是策略，而是决策片段
- 一个Gene只负责一个微观决策
- Gene之间可以组合但不知道彼此存在
- Gene没有"好坏"，只有"适配环境与否"
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
import hashlib
import json
import random
import copy

if TYPE_CHECKING:
    from core.environment import EnvironmentState


class GeneType(Enum):
    """基因类型枚举"""
    SIGNAL = auto()         # 信号基因：产生买卖信号
    FILTER = auto()         # 过滤基因：过滤/确认信号
    POSITION = auto()       # 仓位基因：决定仓位比例
    EXIT = auto()           # 退出基因：决定退出条件
    RISK = auto()           # 风险基因：风险感知与响应
    TIMING = auto()         # 时机基因：决定何时行动
    DORMANT = auto()        # 休眠基因：特定环境才表达


class SignalDirection(Enum):
    """信号方向"""
    LONG = 1        # 做多
    SHORT = -1      # 做空
    NEUTRAL = 0     # 中性/无信号


@dataclass
class GeneExpression:
    """
    基因表达结果
    
    基因表达后产生的输出，可以是信号、过滤、仓位建议等
    """
    gene_id: str
    gene_type: GeneType
    expressed: bool             # 是否成功表达
    output: Any                 # 表达输出
    confidence: float = 1.0     # 表达置信度 [0, 1]
    energy_cost: float = 1.0    # 表达消耗的能量
    
    def __repr__(self) -> str:
        if not self.expressed:
            return f"GeneExpression({self.gene_type.name}: NOT_EXPRESSED)"
        return f"GeneExpression({self.gene_type.name}: {self.output}, conf={self.confidence:.2f})"


@dataclass
class Gene(ABC):
    """
    基因基类 - 最小可遗传单元
    
    设计原则：
    - 一个Gene只负责一个微观决策
    - Gene之间可以组合但不知道彼此存在
    - Gene没有"好坏"，只有"适配环境与否"
    """
    
    gene_type: GeneType
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # 表达条件（何时激活）
    expression_threshold: float = 0.5
    
    # 元数据
    creation_tick: int = 0
    mutation_count: int = 0
    lineage_depth: int = 0  # 从原始基因经过多少代
    
    _gene_id: str = field(default="", init=False)
    
    def __post_init__(self):
        """初始化后计算基因ID"""
        self._gene_id = self._compute_gene_id()
    
    @property
    def gene_id(self) -> str:
        """基因的唯一标识（基于内容的哈希）"""
        if not self._gene_id:
            self._gene_id = self._compute_gene_id()
        return self._gene_id
    
    def _compute_gene_id(self) -> str:
        """计算基因ID"""
        content = {
            "type": self.gene_type.name,
            "params": self.parameters,
            "class": self.__class__.__name__,
        }
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    @abstractmethod
    def can_express(self, env_state: EnvironmentState) -> bool:
        """
        检查在当前环境下是否可以表达
        
        Args:
            env_state: 当前环境状态
            
        Returns:
            是否可以表达
        """
        pass
    
    @abstractmethod
    def express(self, env_state: EnvironmentState) -> GeneExpression:
        """
        基因表达
        
        Args:
            env_state: 当前环境状态
            
        Returns:
            表达结果
        """
        pass
    
    @abstractmethod
    def mutate(self, mutation_rate: float = 0.1) -> Gene:
        """
        基因突变
        
        Args:
            mutation_rate: 突变强度
            
        Returns:
            突变后的新基因
        """
        pass
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """获取参数值"""
        return self.parameters.get(key, default)
    
    def set_parameter(self, key: str, value: Any) -> None:
        """设置参数值（会使gene_id失效，需要重新计算）"""
        self.parameters[key] = value
        self._gene_id = ""  # 标记需要重新计算
    
    def clone(self) -> Gene:
        """
        克隆基因（深拷贝）
        """
        return copy.deepcopy(self)
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "gene_id": self.gene_id,
            "gene_type": self.gene_type.name,
            "gene_class": self.__class__.__name__,
            "parameters": self.parameters,
            "expression_threshold": self.expression_threshold,
            "creation_tick": self.creation_tick,
            "mutation_count": self.mutation_count,
            "lineage_depth": self.lineage_depth,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.gene_id[:8]}, type={self.gene_type.name})"
    
    def __hash__(self) -> int:
        return hash(self.gene_id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Gene):
            return False
        return self.gene_id == other.gene_id


# ============================================================
# 具体基因实现示例
# ============================================================

@dataclass
class SimpleSignalGene(Gene):
    """
    简单信号基因
    
    基于单一指标产生买卖信号
    """
    
    gene_type: GeneType = field(default=GeneType.SIGNAL)
    
    def __post_init__(self):
        # 设置默认参数
        if "indicator" not in self.parameters:
            self.parameters["indicator"] = "price_momentum"
        if "threshold_long" not in self.parameters:
            self.parameters["threshold_long"] = 0.02
        if "threshold_short" not in self.parameters:
            self.parameters["threshold_short"] = -0.02
        if "lookback" not in self.parameters:
            self.parameters["lookback"] = 20
        super().__post_init__()
    
    def can_express(self, env_state: EnvironmentState) -> bool:
        """检查是否有足够的历史数据"""
        lookback = self.parameters.get("lookback", 20)
        return len(env_state.price_history) >= lookback
    
    def express(self, env_state: EnvironmentState) -> GeneExpression:
        """表达信号"""
        if not self.can_express(env_state):
            return GeneExpression(
                gene_id=self.gene_id,
                gene_type=self.gene_type,
                expressed=False,
                output=SignalDirection.NEUTRAL,
            )
        
        # 计算动量
        lookback = self.parameters["lookback"]
        prices = env_state.price_history[-lookback:]
        momentum = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0
        
        # 产生信号
        threshold_long = self.parameters["threshold_long"]
        threshold_short = self.parameters["threshold_short"]
        
        if momentum > threshold_long:
            signal = SignalDirection.LONG
            confidence = min(1.0, momentum / threshold_long)
        elif momentum < threshold_short:
            signal = SignalDirection.SHORT
            confidence = min(1.0, abs(momentum / threshold_short))
        else:
            signal = SignalDirection.NEUTRAL
            confidence = 0.0
        
        return GeneExpression(
            gene_id=self.gene_id,
            gene_type=self.gene_type,
            expressed=True,
            output=signal,
            confidence=confidence,
            energy_cost=1.0,
        )
    
    def mutate(self, mutation_rate: float = 0.1) -> Gene:
        """突变参数"""
        new_gene = self.clone()
        new_gene.mutation_count += 1
        new_gene.lineage_depth += 1
        
        # 参数突变
        for key in ["threshold_long", "threshold_short"]:
            if random.random() < mutation_rate:
                current = new_gene.parameters[key]
                change = random.gauss(0, abs(current) * 0.2)
                new_gene.parameters[key] = current + change
        
        if random.random() < mutation_rate:
            current_lookback = new_gene.parameters["lookback"]
            change = random.randint(-5, 5)
            new_gene.parameters["lookback"] = max(5, current_lookback + change)
        
        new_gene._gene_id = ""  # 重新计算ID
        return new_gene


@dataclass
class SimplePositionGene(Gene):
    """
    简单仓位基因
    
    决定交易的仓位比例
    """
    
    gene_type: GeneType = field(default=GeneType.POSITION)
    
    def __post_init__(self):
        if "base_position" not in self.parameters:
            self.parameters["base_position"] = 0.1  # 10%仓位
        if "confidence_scale" not in self.parameters:
            self.parameters["confidence_scale"] = True  # 是否根据信心调整
        if "max_position" not in self.parameters:
            self.parameters["max_position"] = 0.5  # 最大仓位50%
        super().__post_init__()
    
    def can_express(self, env_state: EnvironmentState) -> bool:
        """仓位基因总是可以表达"""
        return True
    
    def express(self, env_state: EnvironmentState) -> GeneExpression:
        """计算仓位"""
        base = self.parameters["base_position"]
        max_pos = self.parameters["max_position"]
        
        # 简单实现：返回基础仓位
        position = min(base, max_pos)
        
        return GeneExpression(
            gene_id=self.gene_id,
            gene_type=self.gene_type,
            expressed=True,
            output=position,
            confidence=1.0,
            energy_cost=0.5,
        )
    
    def mutate(self, mutation_rate: float = 0.1) -> Gene:
        """突变参数"""
        new_gene = self.clone()
        new_gene.mutation_count += 1
        new_gene.lineage_depth += 1
        
        if random.random() < mutation_rate:
            current = new_gene.parameters["base_position"]
            change = random.gauss(0, 0.05)
            new_gene.parameters["base_position"] = max(0.01, min(0.5, current + change))
        
        new_gene._gene_id = ""
        return new_gene


@dataclass
class SimpleRiskGene(Gene):
    """
    简单风险基因
    
    决定风险容忍度和止损策略
    """
    
    gene_type: GeneType = field(default=GeneType.RISK)
    
    def __post_init__(self):
        if "stop_loss" not in self.parameters:
            self.parameters["stop_loss"] = 0.05  # 5%止损
        if "risk_tolerance" not in self.parameters:
            self.parameters["risk_tolerance"] = 0.5  # 风险容忍度
        if "volatility_sensitivity" not in self.parameters:
            self.parameters["volatility_sensitivity"] = 1.0
        super().__post_init__()
    
    def can_express(self, env_state: EnvironmentState) -> bool:
        return True
    
    def express(self, env_state: EnvironmentState) -> GeneExpression:
        """评估风险"""
        stop_loss = self.parameters["stop_loss"]
        tolerance = self.parameters["risk_tolerance"]
        
        # 计算当前波动率影响
        vol_sensitivity = self.parameters["volatility_sensitivity"]
        current_vol = env_state.volatility if hasattr(env_state, 'volatility') else 0.02
        
        # 根据波动率调整风险
        adjusted_stop = stop_loss * (1 + current_vol * vol_sensitivity)
        
        return GeneExpression(
            gene_id=self.gene_id,
            gene_type=self.gene_type,
            expressed=True,
            output={
                "stop_loss": adjusted_stop,
                "risk_tolerance": tolerance,
                "current_risk_level": current_vol * vol_sensitivity,
            },
            confidence=1.0,
            energy_cost=0.5,
        )
    
    def mutate(self, mutation_rate: float = 0.1) -> Gene:
        """突变参数"""
        new_gene = self.clone()
        new_gene.mutation_count += 1
        new_gene.lineage_depth += 1
        
        if random.random() < mutation_rate:
            current = new_gene.parameters["stop_loss"]
            change = random.gauss(0, 0.01)
            new_gene.parameters["stop_loss"] = max(0.01, min(0.2, current + change))
        
        if random.random() < mutation_rate:
            current = new_gene.parameters["risk_tolerance"]
            change = random.gauss(0, 0.1)
            new_gene.parameters["risk_tolerance"] = max(0.1, min(1.0, current + change))
        
        new_gene._gene_id = ""
        return new_gene


@dataclass  
class DormantGene(Gene):
    """
    休眠基因
    
    在特定环境条件下才会表达
    例如：只在高波动/危机时期才激活的防御机制
    """
    
    gene_type: GeneType = field(default=GeneType.DORMANT)
    wrapped_gene: Optional[Gene] = None  # 包装的实际基因
    
    def __post_init__(self):
        if "activation_condition" not in self.parameters:
            self.parameters["activation_condition"] = "high_volatility"
        if "activation_threshold" not in self.parameters:
            self.parameters["activation_threshold"] = 0.05  # 5%波动率
        super().__post_init__()
    
    def can_express(self, env_state: EnvironmentState) -> bool:
        """只在特定条件下表达"""
        condition = self.parameters["activation_condition"]
        threshold = self.parameters["activation_threshold"]
        
        if condition == "high_volatility":
            vol = getattr(env_state, 'volatility', 0)
            return vol > threshold
        elif condition == "crisis":
            # 检查是否处于危机模式
            regime = getattr(env_state, 'regime', None)
            return regime == "CRISIS" if regime else False
        elif condition == "drawdown":
            # 检查回撤
            drawdown = getattr(env_state, 'drawdown', 0)
            return drawdown > threshold
        
        return False
    
    def express(self, env_state: EnvironmentState) -> GeneExpression:
        """表达包装的基因"""
        if not self.can_express(env_state):
            return GeneExpression(
                gene_id=self.gene_id,
                gene_type=self.gene_type,
                expressed=False,
                output=None,
            )
        
        if self.wrapped_gene:
            return self.wrapped_gene.express(env_state)
        
        # 默认行为：发出风险警告
        return GeneExpression(
            gene_id=self.gene_id,
            gene_type=self.gene_type,
            expressed=True,
            output={"alert": "dormant_gene_activated", "condition": self.parameters["activation_condition"]},
            confidence=1.0,
            energy_cost=2.0,  # 休眠基因激活消耗更多能量
        )
    
    def mutate(self, mutation_rate: float = 0.1) -> Gene:
        """突变"""
        new_gene = self.clone()
        new_gene.mutation_count += 1
        new_gene.lineage_depth += 1
        
        if random.random() < mutation_rate:
            current = new_gene.parameters["activation_threshold"]
            change = random.gauss(0, 0.01)
            new_gene.parameters["activation_threshold"] = max(0.01, current + change)
        
        # 也突变包装的基因
        if new_gene.wrapped_gene and random.random() < mutation_rate:
            new_gene.wrapped_gene = new_gene.wrapped_gene.mutate(mutation_rate)
        
        new_gene._gene_id = ""
        return new_gene
