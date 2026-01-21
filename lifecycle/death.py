"""
死亡机制 - 进化的必要条件

核心原则:
- 死亡是进化的必要条件
- 没有死亡，就没有选择压力
- 死亡是自然的，不是惩罚
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from core.agent import Agent, DeathCause
    from core.environment import Environment
    from core.dna import DNA


@dataclass
class DeathCertificate:
    """
    死亡证明
    
    记录Agent死亡的所有信息，用于后续分析
    """
    
    agent_id: str
    death_tick: int
    death_time: datetime
    cause: DeathCause
    
    # 死亡时的状态
    final_capital: float
    final_position_value: float
    
    # 生命统计
    birth_tick: int
    lifespan: int  # 存活的tick数
    generation: int
    offspring_count: int
    
    # 交易统计
    total_trades: int
    winning_trades: int
    total_pnl: float
    max_drawdown: float
    
    # DNA存档（用于研究和可能的"复活"实验）
    dna_archive: Dict[str, Any] = field(default_factory=dict)
    
    # 谱系信息
    parent_ids: List[str] = field(default_factory=list)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        return {
            "agent_id": self.agent_id[:8],
            "cause": self.cause.name,
            "lifespan": self.lifespan,
            "generation": self.generation,
            "final_capital": self.final_capital,
            "total_pnl": self.total_pnl,
            "offspring": self.offspring_count,
            "win_rate": self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
        }
    
    def __repr__(self) -> str:
        return (
            f"DeathCertificate(agent={self.agent_id[:8]}, "
            f"cause={self.cause.name}, "
            f"lifespan={self.lifespan}, "
            f"pnl={self.total_pnl:.2f})"
        )


class DeathMechanism:
    """
    死亡机制
    
    死亡是进化的必要条件。
    没有fitness function，只有死亡判定。
    
    死亡类型：
    1. STARVATION: 资本耗尽（饿死）
    2. PREDATION: 风险爆仓（被"捕食"）
    3. METABOLIC_FAILURE: 代谢失败（无法维持生存成本）
    4. SENESCENCE: 自然寿命（可选）
    5. EXTINCTION_EVENT: 环境剧变（大灭绝）
    """
    
    # 生存阈值
    SURVIVAL_THRESHOLD: float = 100.0  # 资本低于此值视为濒危
    DYING_THRESHOLD: int = 10          # 连续N个tick无法支付代谢即死亡
    MAX_LIFESPAN: Optional[int] = None # 最大寿命（None表示无限）
    
    def __init__(
        self,
        survival_threshold: float = 100.0,
        dying_threshold: int = 10,
        max_lifespan: Optional[int] = None,
        enable_natural_death: bool = False,
    ):
        """
        初始化死亡机制
        
        Args:
            survival_threshold: 生存资本阈值
            dying_threshold: 濒死阈值
            max_lifespan: 最大寿命（None表示无限）
            enable_natural_death: 是否启用自然死亡
        """
        self.survival_threshold = survival_threshold
        self.dying_threshold = dying_threshold
        self.max_lifespan = max_lifespan
        self.enable_natural_death = enable_natural_death
    
    def check_death(
        self, 
        agent: Agent, 
        environment: Environment,
        current_tick: int
    ) -> Optional[DeathCertificate]:
        """
        检查Agent是否应该死亡
        
        Args:
            agent: 要检查的Agent
            environment: 当前环境
            current_tick: 当前时刻
            
        Returns:
            死亡证明（如果死亡），None表示存活
        """
        from core.agent import DeathCause, AgentState
        
        # 已经死亡的不再检查
        if agent.state == AgentState.DEAD:
            return None
        
        death_cause: Optional[DeathCause] = None
        
        # 1. 饿死检查：资本低于生存阈值
        if agent.resources.capital < self.survival_threshold:
            death_cause = DeathCause.STARVATION
        
        # 2. 风险死亡检查
        elif agent.resources.is_risk_exhausted() and not agent.position.is_empty():
            # 有持仓但风险预算耗尽
            death_cause = DeathCause.PREDATION
        
        # 3. 代谢死亡检查（在Agent内部处理）
        elif agent.state == AgentState.DYING and agent._dying_counter >= self.dying_threshold:
            death_cause = DeathCause.METABOLIC_FAILURE
        
        # 4. 自然寿命检查
        elif self.enable_natural_death and self.max_lifespan is not None:
            age = agent.get_age(current_tick)
            if age >= self.max_lifespan:
                death_cause = DeathCause.SENESCENCE
        
        # 5. 环境剧变检查
        # 这通常由外部事件触发，不在常规检查中
        
        if death_cause is not None:
            return self._create_death_certificate(agent, death_cause, current_tick)
        
        return None
    
    def execute_death(
        self, 
        agent: Agent, 
        certificate: DeathCertificate
    ) -> None:
        """
        执行死亡
        
        Args:
            agent: 要处死的Agent
            certificate: 死亡证明
        """
        from core.agent import AgentState
        
        agent.state = AgentState.DEAD
        agent.death_tick = certificate.death_tick
        agent.death_cause = certificate.cause
        
        # 清空持仓（死亡时强制平仓）
        agent.position = agent.position.__class__()  # 重置为空持仓
    
    def trigger_extinction_event(
        self, 
        agents: List[Agent], 
        survival_rate: float,
        current_tick: int
    ) -> List[DeathCertificate]:
        """
        触发大灭绝事件
        
        Args:
            agents: Agent列表
            survival_rate: 存活率 [0, 1]
            current_tick: 当前时刻
            
        Returns:
            死亡证明列表
        """
        from core.agent import DeathCause, AgentState
        import random
        
        certificates = []
        alive_agents = [a for a in agents if a.is_alive]
        
        for agent in alive_agents:
            if random.random() > survival_rate:
                # 死于大灭绝
                cert = self._create_death_certificate(
                    agent, 
                    DeathCause.EXTINCTION_EVENT, 
                    current_tick
                )
                self.execute_death(agent, cert)
                certificates.append(cert)
        
        return certificates
    
    def _create_death_certificate(
        self, 
        agent: Agent, 
        cause: DeathCause,
        death_tick: int
    ) -> DeathCertificate:
        """创建死亡证明"""
        return DeathCertificate(
            agent_id=agent.agent_id,
            death_tick=death_tick,
            death_time=datetime.now(),
            cause=cause,
            final_capital=agent.resources.capital,
            final_position_value=agent.position.unrealized_pnl,
            birth_tick=agent.birth_tick,
            lifespan=death_tick - agent.birth_tick,
            generation=agent.lineage.generation,
            offspring_count=agent.offspring_count,
            total_trades=agent.stats.total_trades,
            winning_trades=agent.stats.winning_trades,
            total_pnl=agent.stats.total_pnl,
            max_drawdown=agent.stats.max_drawdown,
            dna_archive=agent.dna.to_dict(),
            parent_ids=agent.lineage.parent_ids,
        )
    
    def get_mortality_stats(
        self, 
        certificates: List[DeathCertificate]
    ) -> Dict[str, Any]:
        """
        获取死亡统计
        
        Args:
            certificates: 死亡证明列表
            
        Returns:
            统计数据
        """
        if not certificates:
            return {"total_deaths": 0}
        
        # 按死因分类
        causes = {}
        for cert in certificates:
            cause_name = cert.cause.name
            causes[cause_name] = causes.get(cause_name, 0) + 1
        
        # 计算平均寿命
        lifespans = [c.lifespan for c in certificates]
        avg_lifespan = sum(lifespans) / len(lifespans)
        
        # 计算平均盈亏
        pnls = [c.total_pnl for c in certificates]
        avg_pnl = sum(pnls) / len(pnls)
        
        return {
            "total_deaths": len(certificates),
            "causes": causes,
            "avg_lifespan": avg_lifespan,
            "max_lifespan": max(lifespans),
            "min_lifespan": min(lifespans),
            "avg_pnl": avg_pnl,
            "avg_generation": sum(c.generation for c in certificates) / len(certificates),
        }
