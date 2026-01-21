"""
Agent系统 - 资源受限的生命体

核心原则:
- Agent是资源受限实体
- 资本有限且可耗尽
- 注意力有限
- 风险承受有上限
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import uuid

from resources import AgentResources
from core.dna import DNA, Phenotype, TradeAction
from core.gene import SignalDirection

if TYPE_CHECKING:
    from environment import Environment, EnvironmentState, Order, ExecutionResult


class AgentState(Enum):
    """Agent生命状态"""
    EMBRYO = auto()     # 胚胎期（刚出生）
    ALIVE = auto()      # 存活
    DYING = auto()      # 濒死
    DEAD = auto()       # 死亡


class DeathCause(Enum):
    """死亡原因"""
    STARVATION = auto()         # 资本耗尽
    PREDATION = auto()          # 风险爆仓
    METABOLIC_FAILURE = auto()  # 代谢失败
    SENESCENCE = auto()         # 自然寿命
    EXTINCTION_EVENT = auto()   # 环境剧变


@dataclass
class Lineage:
    """血统记录"""
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0
    birth_tick: int = 0
    
    @classmethod
    def from_parents(cls, parent1: Agent, parent2: Optional[Agent] = None) -> Lineage:
        """从父代创建血统"""
        parents = [parent1.agent_id]
        if parent2:
            parents.append(parent2.agent_id)
        
        max_gen = parent1.lineage.generation
        if parent2:
            max_gen = max(max_gen, parent2.lineage.generation)
        
        return cls(
            parent_ids=parents,
            generation=max_gen + 1,
        )


@dataclass
class Position:
    """持仓"""
    symbol: str = "default"
    direction: int = 0  # 1=多, -1=空, 0=无
    quantity: float = 0.0
    entry_price: float = 0.0
    entry_tick: int = 0
    unrealized_pnl: float = 0.0
    
    def update_pnl(self, current_price: float) -> float:
        """更新未实现盈亏"""
        if self.quantity == 0:
            self.unrealized_pnl = 0.0
        else:
            price_diff = current_price - self.entry_price
            self.unrealized_pnl = price_diff * self.quantity * self.direction
        return self.unrealized_pnl
    
    def is_empty(self) -> bool:
        return self.quantity == 0 or self.direction == 0


@dataclass
class AgentStats:
    """Agent统计数据"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_capital: float = 0.0
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades


class Agent:
    """
    Agent - 资源受限的生命体
    
    Agent是一个资源受限实体，其行为由DNA决定，
    但受到资源约束的限制。
    
    不可违背的约束：
    - 资本有限且可耗尽
    - 注意力有限
    - 算力有限
    - 风险承受有上限
    """
    
    def __init__(
        self,
        dna: DNA,
        resources: Optional[AgentResources] = None,
        lineage: Optional[Lineage] = None,
        birth_tick: int = 0,
    ):
        """
        创建Agent
        
        Args:
            dna: Agent的DNA
            resources: 初始资源（默认使用标准资源）
            lineage: 血统记录
            birth_tick: 出生时刻
        """
        self.agent_id: str = str(uuid.uuid4())
        self.dna: DNA = dna
        self.resources: AgentResources = resources or AgentResources()
        self.lineage: Lineage = lineage or Lineage()
        self.lineage.birth_tick = birth_tick
        
        # 状态
        self.state: AgentState = AgentState.EMBRYO
        self.birth_tick: int = birth_tick
        self.death_tick: Optional[int] = None
        self.death_cause: Optional[DeathCause] = None
        
        # 持仓
        self.position: Position = Position()
        
        # 繁殖相关
        self.reproductive_energy: float = 0.0  # 繁殖能量（需要积累）
        self.offspring_count: int = 0
        self.reproduction_cooldown: int = 0  # 繁殖冷却
        
        # 濒死计数器
        self._dying_counter: int = 0
        self._dying_threshold: int = 10  # 连续N个tick无法支付代谢即死亡
        
        # 统计
        self.stats: AgentStats = AgentStats()
        self.stats.peak_capital = self.resources.capital
        
        # 行动历史（有限长度）
        self.action_history: List[Dict[str, Any]] = []
        self._max_history: int = 100
    
    @property
    def age(self) -> int:
        """年龄（存活的tick数）"""
        if self.death_tick is not None:
            return self.death_tick - self.birth_tick
        return 0  # 需要外部提供当前tick
    
    def get_age(self, current_tick: int) -> int:
        """获取年龄"""
        if self.death_tick is not None:
            return self.death_tick - self.birth_tick
        return current_tick - self.birth_tick
    
    @property
    def is_alive(self) -> bool:
        """是否存活"""
        return self.state in [AgentState.EMBRYO, AgentState.ALIVE, AgentState.DYING]
    
    def activate(self) -> None:
        """激活Agent（从胚胎到存活）"""
        if self.state == AgentState.EMBRYO:
            self.state = AgentState.ALIVE
    
    def live_one_tick(
        self, 
        environment: Environment, 
        current_tick: int
    ) -> Optional[Order]:
        """
        Agent的一个生命周期刻度
        
        顺序不可变：
        1. 激活检查
        2. 支付生存成本（代谢）
        3. 感知环境
        4. 基因表达 → 产生行为
        5. 资源约束检查
        6. 更新繁殖能量
        7. 死亡检查
        
        Args:
            environment: 市场环境
            current_tick: 当前时刻
            
        Returns:
            交易订单（如果有的话）
        """
        from environment import Order
        
        # 0. 死亡Agent不行动
        if not self.is_alive:
            return None
        
        # 1. 激活检查
        if self.state == AgentState.EMBRYO:
            self.activate()
        
        # 2. 支付代谢成本
        if not self._pay_metabolism():
            self._dying_counter += 1
            if self._dying_counter >= self._dying_threshold:
                self._die(DeathCause.METABOLIC_FAILURE, current_tick)
                return None
            self.state = AgentState.DYING
        else:
            self._dying_counter = 0
            if self.state == AgentState.DYING:
                self.state = AgentState.ALIVE
        
        # 3. 感知环境（受注意力限制）
        env_state = self._perceive(environment)
        
        # 4. 更新持仓盈亏
        self._update_position(env_state.current_price, current_tick)
        
        # 5. 基因表达
        phenotype = self.dna.express(env_state)
        
        # 6. 消耗表达能量
        if not self.resources.consume_energy(phenotype.total_energy_cost):
            # 能量不足，无法行动
            return None
        
        # 7. 将表型转换为行动
        action = phenotype.to_action()
        order = None
        
        if action:
            order = self._create_order(action, env_state)
        
        # 8. 资源再生
        self.resources.regenerate()
        
        # 9. 更新繁殖能量
        self._update_reproductive_energy()
        
        # 10. 冷却时间递减
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1
        
        # 11. 死亡检查
        self._death_check(current_tick)
        
        # 记录行动
        self._record_action(current_tick, phenotype, action)
        
        return order
    
    def _pay_metabolism(self) -> bool:
        """支付代谢成本"""
        return self.resources.pay_metabolism()
    
    def _perceive(self, environment: Environment) -> EnvironmentState:
        """
        感知环境
        
        受注意力限制，只能看到有限的信息
        """
        visible_history = min(100, self.resources.attention_units * 20)
        return environment.get_state(visible_history)
    
    def _update_position(self, current_price: float, current_tick: int) -> None:
        """更新持仓状态"""
        if not self.position.is_empty():
            pnl = self.position.update_pnl(current_price)
            
            # 检查止损
            if pnl < -self.resources.capital * 0.1:  # 损失超过10%资本
                # 强制平仓
                self._close_position(current_price, current_tick, forced=True)
    
    def _create_order(
        self, 
        action: TradeAction, 
        env_state: EnvironmentState
    ) -> Optional[Order]:
        """
        创建订单
        
        Args:
            action: 交易行动
            env_state: 环境状态
            
        Returns:
            订单
        """
        from environment import Order
        
        # 检查风险预算
        required_risk = action.position_size * self.resources.capital * action.stop_loss
        if not self.resources.consume_risk_budget(required_risk):
            return None
        
        # 计算实际数量
        position_value = self.resources.capital * action.position_size
        quantity = position_value / env_state.current_price
        
        # 创建订单
        direction = 1 if action.direction == SignalDirection.LONG else -1
        
        return Order(
            agent_id=self.agent_id,
            direction=direction,
            quantity=quantity,
        )
    
    def process_execution(
        self, 
        result: ExecutionResult, 
        current_tick: int
    ) -> None:
        """
        处理订单执行结果
        
        Args:
            result: 执行结果
            current_tick: 当前时刻
        """
        if not result.executed:
            return
        
        # 支付成本
        total_cost = result.total_cost
        
        if result.order.direction > 0:  # 买入
            # 开多仓
            if not self.resources.consume_capital(total_cost):
                return  # 资金不足
            
            self.position = Position(
                direction=1,
                quantity=result.executed_quantity,
                entry_price=result.executed_price,
                entry_tick=current_tick,
            )
        else:  # 卖出
            if not self.position.is_empty():
                # 平仓
                self._close_position(result.executed_price, current_tick)
            else:
                # 开空仓
                self.position = Position(
                    direction=-1,
                    quantity=result.executed_quantity,
                    entry_price=result.executed_price,
                    entry_tick=current_tick,
                )
        
        # 更新统计
        self.stats.total_trades += 1
    
    def _close_position(
        self, 
        close_price: float, 
        current_tick: int,
        forced: bool = False
    ) -> None:
        """平仓"""
        if self.position.is_empty():
            return
        
        # 计算盈亏
        price_diff = close_price - self.position.entry_price
        pnl = price_diff * self.position.quantity * self.position.direction
        
        # 平仓时的资本处理
        # 对于多仓（direction=1）：开仓时支付了entry_price*quantity，平仓时卖出获得close_price*quantity
        # close_price * quantity 总是正数（价格和数量都是正数），可以安全使用add_capital
        capital_returned = close_price * self.position.quantity
        self.resources.add_capital(capital_returned)
        
        # 恢复风险预算
        self.resources.restore_risk_budget(
            self.position.quantity * self.position.entry_price * 0.05
        )
        
        # 更新统计
        self.stats.total_pnl += pnl
        if pnl > 0:
            self.stats.winning_trades += 1
        else:
            self.stats.losing_trades += 1
        
        # 更新峰值
        if self.resources.capital > self.stats.peak_capital:
            self.stats.peak_capital = self.resources.capital
        
        # 计算回撤
        current_drawdown = (self.stats.peak_capital - self.resources.capital) / self.stats.peak_capital
        if current_drawdown > self.stats.max_drawdown:
            self.stats.max_drawdown = current_drawdown
        
        # 清空持仓
        self.position = Position()
    
    def _update_reproductive_energy(self) -> None:
        """
        更新繁殖能量
        
        繁殖能量基于：
        1. 盈利能力
        2. 生存时间
        3. 资源效率
        """
        # 基础能量积累
        base_energy = 0.1
        
        # 盈利加成
        if self.stats.total_pnl > 0:
            profit_bonus = min(1.0, self.stats.total_pnl / self.resources.capital)
            base_energy += profit_bonus * 0.5
        
        # 存活加成（越长越多）
        survival_bonus = min(0.5, self._dying_counter == 0 and 0.1 or 0)
        base_energy += survival_bonus
        
        self.reproductive_energy += base_energy
    
    def can_reproduce(self) -> bool:
        """
        检查是否可以繁殖
        
        条件：
        1. 存活状态
        2. 繁殖能量充足
        3. 冷却时间结束
        4. 资源充足
        """
        if not self.is_alive:
            return False
        
        if self.reproductive_energy < 100:
            return False
        
        if self.reproduction_cooldown > 0:
            return False
        
        # 需要有足够资本分给后代
        if self.resources.capital < 2000:  # 至少2000才能繁殖
            return False
        
        return True
    
    def prepare_reproduction(self) -> Tuple[DNA, AgentResources]:
        """
        准备繁殖材料
        
        Returns:
            (DNA, Resources) 用于创建后代
        """
        # 消耗繁殖能量
        self.reproductive_energy -= 100
        
        # 设置冷却
        self.reproduction_cooldown = 50  # 50 tick冷却
        
        # 增加后代计数
        self.offspring_count += 1
        
        # 分出资源给后代
        child_resources = self.resources.clone(inheritance_ratio=0.3)
        self.resources.capital *= 0.7  # 自己保留70%
        
        return self.dna.clone(), child_resources
    
    def _death_check(self, current_tick: int) -> None:
        """死亡检查"""
        # 1. 破产死亡
        if self.resources.is_bankrupt():
            self._die(DeathCause.STARVATION, current_tick)
            return
        
        # 2. 风险死亡
        if self.resources.is_risk_exhausted() and not self.position.is_empty():
            # 强制平仓后如果仍然资源耗尽
            if self.resources.is_bankrupt():
                self._die(DeathCause.PREDATION, current_tick)
                return
    
    def _die(self, cause: DeathCause, current_tick: int) -> None:
        """死亡"""
        self.state = AgentState.DEAD
        self.death_tick = current_tick
        self.death_cause = cause
        
        # 清空持仓
        self.position = Position()
    
    def _record_action(
        self, 
        tick: int, 
        phenotype: Phenotype, 
        action: Optional[TradeAction]
    ) -> None:
        """记录行动"""
        record = {
            "tick": tick,
            "signal": phenotype.signal.name,
            "confidence": phenotype.signal_confidence,
            "expressed_genes": phenotype.expressed_genes,
            "action": action.direction.name if action else None,
            "position_size": action.position_size if action else 0,
            "capital": self.resources.capital,
            "state": self.state.name,
        }
        
        self.action_history.append(record)
        
        # 限制历史长度
        if len(self.action_history) > self._max_history:
            self.action_history.pop(0)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取Agent摘要"""
        return {
            "agent_id": self.agent_id[:8],
            "dna_id": self.dna.dna_id[:8],
            "state": self.state.name,
            "generation": self.lineage.generation,
            "capital": self.resources.capital,
            "total_pnl": self.stats.total_pnl,
            "win_rate": self.stats.win_rate,
            "offspring": self.offspring_count,
            "reproductive_energy": self.reproductive_energy,
            "death_cause": self.death_cause.name if self.death_cause else None,
        }
    
    def __repr__(self) -> str:
        return (
            f"Agent(id={self.agent_id[:8]}, "
            f"state={self.state.name}, "
            f"gen={self.lineage.generation}, "
            f"capital={self.resources.capital:.2f})"
        )
    
    def __hash__(self) -> int:
        return hash(self.agent_id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Agent):
            return False
        return self.agent_id == other.agent_id
