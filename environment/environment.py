"""
市场环境 - 不可控的外部世界

核心原则:
- 环境不为任何Agent服务
- 环境有自己的演化规律
- 环境可以突变（regime change）
- Agent的行为对环境的影响是有限的
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import random
import math
import numpy as np


class MarketRegime(Enum):
    """市场制度/状态"""
    BULL = auto()           # 牛市
    BEAR = auto()           # 熊市
    SIDEWAYS = auto()       # 震荡
    HIGH_VOLATILITY = auto()  # 高波动
    LOW_VOLATILITY = auto()   # 低波动
    CRISIS = auto()         # 危机
    RECOVERY = auto()       # 复苏


@dataclass
class MarketState:
    """市场状态快照"""
    tick: int
    price: float
    volume: float
    volatility: float
    regime: MarketRegime
    
    # 衍生数据
    price_change: float = 0.0
    price_change_pct: float = 0.0
    
    def __repr__(self) -> str:
        return f"MarketState(t={self.tick}, p={self.price:.2f}, vol={self.volatility:.4f}, {self.regime.name})"


@dataclass
class EnvironmentState:
    """
    环境状态 - 传递给Agent用于决策
    
    这是Agent能"看到"的环境信息（可能是有限的）
    """
    
    current_tick: int
    current_price: float
    price_history: List[float] = field(default_factory=list)
    volume_history: List[float] = field(default_factory=list)
    volatility: float = 0.02
    regime: Optional[MarketRegime] = None
    
    # 限制：Agent不能看到所有信息
    visible_history_length: int = 100  # 只能看到最近N个tick的历史
    
    # 市场深度信息（可选）
    bid_price: float = 0.0
    ask_price: float = 0.0
    spread: float = 0.0
    
    # 计算属性
    @property
    def drawdown(self) -> float:
        """当前回撤"""
        if not self.price_history:
            return 0.0
        peak = max(self.price_history)
        return (peak - self.current_price) / peak if peak > 0 else 0.0
    
    @property
    def momentum(self) -> float:
        """短期动量"""
        if len(self.price_history) < 2:
            return 0.0
        return (self.current_price - self.price_history[-min(20, len(self.price_history))]) / self.price_history[-min(20, len(self.price_history))]


@dataclass
class Order:
    """订单"""
    agent_id: str
    direction: int  # 1=买, -1=卖
    quantity: float
    price: Optional[float] = None  # None表示市价单
    order_type: str = "MARKET"  # MARKET, LIMIT
    
    def __repr__(self) -> str:
        dir_str = "BUY" if self.direction > 0 else "SELL"
        return f"Order({dir_str}, qty={self.quantity:.4f}, type={self.order_type})"


@dataclass
class ExecutionResult:
    """订单执行结果"""
    order: Order
    executed: bool
    executed_price: float = 0.0
    executed_quantity: float = 0.0
    slippage: float = 0.0
    commission: float = 0.0
    
    @property
    def total_cost(self) -> float:
        """总成本（含滑点和手续费）"""
        return self.executed_price * self.executed_quantity + self.slippage + self.commission
    
    def __repr__(self) -> str:
        if not self.executed:
            return f"ExecutionResult(FAILED)"
        return f"ExecutionResult(price={self.executed_price:.2f}, qty={self.executed_quantity:.4f}, slip={self.slippage:.4f})"


class Environment:
    """
    市场环境
    
    这是一个独立于所有Agent的外生系统。
    环境有自己的演化规律，不会为任何Agent的期望而改变。
    """
    
    def __init__(
        self,
        initial_price: float = 100.0,
        base_volatility: float = 0.02,
        tick_size: float = 0.01,
        commission_rate: float = 0.001,
        slippage_factor: float = 0.1,
    ):
        """
        初始化市场环境
        
        Args:
            initial_price: 初始价格
            base_volatility: 基础波动率
            tick_size: 最小价格变动
            commission_rate: 手续费率
            slippage_factor: 滑点因子
        """
        self.world_time: int = 0
        self.current_price: float = initial_price
        self.base_volatility: float = base_volatility
        self.current_volatility: float = base_volatility
        self.tick_size: float = tick_size
        self.commission_rate: float = commission_rate
        self.slippage_factor: float = slippage_factor
        
        # 历史记录
        self.price_history: List[float] = [initial_price]
        self.volume_history: List[float] = [0.0]
        self.volatility_history: List[float] = [base_volatility]
        
        # 市场制度
        self.current_regime: MarketRegime = MarketRegime.SIDEWAYS
        self.regime_history: List[Tuple[int, MarketRegime]] = [(0, MarketRegime.SIDEWAYS)]
        
        # 制度转换概率矩阵
        self._regime_transition_probs: Dict[MarketRegime, Dict[MarketRegime, float]] = {
            MarketRegime.BULL: {
                MarketRegime.BULL: 0.9,
                MarketRegime.SIDEWAYS: 0.05,
                MarketRegime.BEAR: 0.03,
                MarketRegime.CRISIS: 0.02,
            },
            MarketRegime.BEAR: {
                MarketRegime.BEAR: 0.85,
                MarketRegime.SIDEWAYS: 0.05,
                MarketRegime.BULL: 0.02,
                MarketRegime.CRISIS: 0.05,
                MarketRegime.RECOVERY: 0.03,
            },
            MarketRegime.SIDEWAYS: {
                MarketRegime.SIDEWAYS: 0.8,
                MarketRegime.BULL: 0.1,
                MarketRegime.BEAR: 0.08,
                MarketRegime.HIGH_VOLATILITY: 0.02,
            },
            MarketRegime.HIGH_VOLATILITY: {
                MarketRegime.HIGH_VOLATILITY: 0.7,
                MarketRegime.CRISIS: 0.15,
                MarketRegime.SIDEWAYS: 0.1,
                MarketRegime.RECOVERY: 0.05,
            },
            MarketRegime.LOW_VOLATILITY: {
                MarketRegime.LOW_VOLATILITY: 0.85,
                MarketRegime.SIDEWAYS: 0.1,
                MarketRegime.HIGH_VOLATILITY: 0.05,
            },
            MarketRegime.CRISIS: {
                MarketRegime.CRISIS: 0.6,
                MarketRegime.RECOVERY: 0.25,
                MarketRegime.BEAR: 0.15,
            },
            MarketRegime.RECOVERY: {
                MarketRegime.RECOVERY: 0.7,
                MarketRegime.BULL: 0.2,
                MarketRegime.SIDEWAYS: 0.1,
            },
        }
        
        # 流动性
        self.liquidity: float = 1.0  # 1.0 = 正常流动性
        
        # 外部数据源（可选）
        self._external_data: Optional[List[float]] = None
        self._external_data_index: int = 0
    
    def set_external_data(self, prices: List[float]) -> None:
        """
        设置外部价格数据
        
        用于使用真实历史数据而非模拟数据
        """
        self._external_data = prices
        self._external_data_index = 0
        if prices:
            self.current_price = prices[0]
            self.price_history = [prices[0]]
    
    def tick(self) -> MarketState:
        """
        环境自身演化一个时刻
        
        这与Agent无关，是市场自身的运动。
        
        Returns:
            当前市场状态
        """
        old_price = self.current_price
        
        # 时间推进
        self.world_time += 1
        
        # 检查制度转换
        self._check_regime_shift()
        
        # 更新波动率
        self._update_volatility()
        
        # 更新价格
        if self._external_data is not None:
            # 使用外部数据
            self._update_price_from_external()
        else:
            # 模拟价格运动
            self._simulate_price_movement()
        
        # 更新流动性
        self._update_liquidity()
        
        # 记录历史
        self.price_history.append(self.current_price)
        self.volatility_history.append(self.current_volatility)
        self.volume_history.append(random.uniform(0.5, 1.5))  # 模拟成交量
        
        # 返回市场状态
        return MarketState(
            tick=self.world_time,
            price=self.current_price,
            volume=self.volume_history[-1],
            volatility=self.current_volatility,
            regime=self.current_regime,
            price_change=self.current_price - old_price,
            price_change_pct=(self.current_price - old_price) / old_price if old_price != 0 else 0,
        )
    
    def _check_regime_shift(self) -> None:
        """检查并执行制度转换"""
        current = self.current_regime
        probs = self._regime_transition_probs.get(current, {})
        
        if not probs:
            return
        
        # 归一化概率
        total = sum(probs.values())
        normalized = {k: v/total for k, v in probs.items()}
        
        # 随机选择新制度
        r = random.random()
        cumsum = 0.0
        for regime, prob in normalized.items():
            cumsum += prob
            if r < cumsum:
                if regime != current:
                    self.current_regime = regime
                    self.regime_history.append((self.world_time, regime))
                break
    
    def _update_volatility(self) -> None:
        """更新波动率"""
        # 根据制度调整波动率
        regime_vol_multiplier = {
            MarketRegime.BULL: 0.8,
            MarketRegime.BEAR: 1.2,
            MarketRegime.SIDEWAYS: 1.0,
            MarketRegime.HIGH_VOLATILITY: 2.5,
            MarketRegime.LOW_VOLATILITY: 0.5,
            MarketRegime.CRISIS: 4.0,
            MarketRegime.RECOVERY: 1.5,
        }
        
        multiplier = regime_vol_multiplier.get(self.current_regime, 1.0)
        
        # 波动率均值回归 + 随机扰动
        target_vol = self.base_volatility * multiplier
        self.current_volatility = 0.95 * self.current_volatility + 0.05 * target_vol
        self.current_volatility *= (1 + random.gauss(0, 0.1))
        self.current_volatility = max(0.001, self.current_volatility)  # 下限
    
    def _simulate_price_movement(self) -> None:
        """模拟价格运动"""
        # 根据制度设置漂移
        regime_drift = {
            MarketRegime.BULL: 0.0005,
            MarketRegime.BEAR: -0.0005,
            MarketRegime.SIDEWAYS: 0.0,
            MarketRegime.HIGH_VOLATILITY: 0.0,
            MarketRegime.LOW_VOLATILITY: 0.0001,
            MarketRegime.CRISIS: -0.002,
            MarketRegime.RECOVERY: 0.001,
        }
        
        drift = regime_drift.get(self.current_regime, 0.0)
        
        # 几何布朗运动
        shock = random.gauss(0, 1)
        price_return = drift + self.current_volatility * shock
        
        self.current_price *= (1 + price_return)
        self.current_price = max(self.tick_size, self.current_price)  # 价格下限
        
        # 四舍五入到tick_size
        self.current_price = round(self.current_price / self.tick_size) * self.tick_size
    
    def _update_price_from_external(self) -> None:
        """从外部数据更新价格"""
        if self._external_data is None:
            return
        
        if self._external_data_index < len(self._external_data):
            self.current_price = self._external_data[self._external_data_index]
            self._external_data_index += 1
        # 如果数据用完，保持最后价格
    
    def _update_liquidity(self) -> None:
        """更新流动性"""
        # 危机时流动性下降
        if self.current_regime == MarketRegime.CRISIS:
            self.liquidity = max(0.2, self.liquidity * 0.95)
        else:
            # 流动性恢复
            self.liquidity = min(1.0, self.liquidity * 1.02)
    
    def get_state(self, visible_history: int = 100) -> EnvironmentState:
        """
        获取环境状态（Agent视角）
        
        Args:
            visible_history: 可见的历史长度
            
        Returns:
            环境状态
        """
        return EnvironmentState(
            current_tick=self.world_time,
            current_price=self.current_price,
            price_history=self.price_history[-visible_history:],
            volume_history=self.volume_history[-visible_history:],
            volatility=self.current_volatility,
            regime=self.current_regime,
            visible_history_length=visible_history,
            bid_price=self.current_price * (1 - self.current_volatility * 0.01),
            ask_price=self.current_price * (1 + self.current_volatility * 0.01),
            spread=self.current_volatility * 0.02,
        )
    
    def execute(self, orders: List[Order]) -> List[ExecutionResult]:
        """
        执行订单
        
        注意：
        - 滑点是真实的
        - 流动性消耗是真实的
        - 你想买的未必买得到
        
        Args:
            orders: 订单列表
            
        Returns:
            执行结果列表
        """
        results = []
        
        # 随机化订单顺序（模拟市场的不确定性）
        shuffled_orders = list(orders)
        random.shuffle(shuffled_orders)
        
        for order in shuffled_orders:
            result = self._execute_single_order(order)
            results.append(result)
        
        return results
    
    def _execute_single_order(self, order: Order) -> ExecutionResult:
        """执行单个订单"""
        # 流动性检查
        if random.random() > self.liquidity:
            # 流动性不足，订单失败
            return ExecutionResult(
                order=order,
                executed=False,
            )
        
        # 计算执行价格（含滑点）
        slippage = self._calculate_slippage(order)
        
        if order.direction > 0:  # 买入
            executed_price = self.current_price + slippage
        else:  # 卖出
            executed_price = self.current_price - slippage
        
        # 部分成交模拟
        fill_ratio = min(1.0, self.liquidity + random.uniform(0, 0.2))
        executed_quantity = order.quantity * fill_ratio
        
        # 手续费
        commission = executed_price * executed_quantity * self.commission_rate
        
        return ExecutionResult(
            order=order,
            executed=True,
            executed_price=executed_price,
            executed_quantity=executed_quantity,
            slippage=slippage * executed_quantity,
            commission=commission,
        )
    
    def _calculate_slippage(self, order: Order) -> float:
        """计算滑点"""
        # 滑点与订单大小和波动率成正比，与流动性成反比
        base_slippage = self.current_price * self.current_volatility * self.slippage_factor
        size_impact = math.sqrt(order.quantity) * 0.01  # 大订单更多滑点
        liquidity_impact = (1 - self.liquidity) * 0.5
        
        total_slippage = base_slippage * (1 + size_impact + liquidity_impact)
        
        # 添加随机性
        total_slippage *= (1 + random.gauss(0, 0.2))
        
        return max(0, total_slippage)
    
    def get_market_summary(self) -> Dict[str, Any]:
        """获取市场摘要"""
        if len(self.price_history) < 2:
            return {
                "tick": self.world_time,
                "price": self.current_price,
                "regime": self.current_regime.name,
            }
        
        return {
            "tick": self.world_time,
            "price": self.current_price,
            "volatility": self.current_volatility,
            "regime": self.current_regime.name,
            "liquidity": self.liquidity,
            "price_change_1": (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2],
            "high_20": max(self.price_history[-20:]) if len(self.price_history) >= 20 else max(self.price_history),
            "low_20": min(self.price_history[-20:]) if len(self.price_history) >= 20 else min(self.price_history),
        }
    
    def __repr__(self) -> str:
        return f"Environment(t={self.world_time}, price={self.current_price:.2f}, regime={self.current_regime.name})"
