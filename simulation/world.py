"""
世界 - 模拟的主循环

核心原则:
- 世界只执行规则，不指定赢家
- 时间不可逆
- 选择通过死亡和繁殖自然发生
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime

from core.world_clock import WorldClock
from environment import Environment, Order
from core.agent import Agent
from simulation.population import Population
from lifecycle.death import DeathMechanism, DeathCertificate
from lifecycle.reproduction import ReproductionEngine
from lifecycle.birth import BirthFactory
from genetics.gene_pool import GenePool
from genetics.mutation import MutationEngine
from storage.event_store import EventStore, EventType
from storage.graveyard import Graveyard
from storage.snapshot import SnapshotStore


@dataclass
class WorldConfig:
    """世界配置"""
    # 种群配置
    initial_population_size: int = 100
    max_population_size: int = 500
    enable_carrying_capacity: bool = True
    
    # 繁殖配置
    reproduction_mode: str = "mixed"  # "asexual", "sexual", "mixed"
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    
    # 死亡配置
    survival_threshold: float = 100.0
    enable_natural_death: bool = False
    max_lifespan: Optional[int] = None
    
    # 存储配置
    snapshot_interval: int = 100
    persist_path: Optional[str] = None
    
    # 环境配置
    initial_price: float = 100.0
    base_volatility: float = 0.02


@dataclass
class TickResult:
    """单个tick的结果"""
    tick: int
    market_state: Dict[str, Any]
    
    # 种群变化
    births: int = 0
    deaths: int = 0
    trades: int = 0
    
    # 当前状态
    alive_count: int = 0
    total_capital: float = 0.0
    
    # 事件
    events: List[str] = field(default_factory=list)


class World:
    """
    世界 - 进化系统的容器
    
    世界是整个模拟的主循环，包含：
    1. 时钟（WorldClock）
    2. 环境（Environment）
    3. 种群（Population）
    4. 各种机制（死亡、繁殖、突变）
    5. 存储系统
    
    核心原则：
    - 世界只执行规则，不干预结果
    - 人类只能制定规则，不能指定赢家
    - 选择通过死亡和繁殖自然发生
    """
    
    def __init__(self, config: Optional[WorldConfig] = None):
        """
        初始化世界
        
        Args:
            config: 世界配置
        """
        self.config = config or WorldConfig()
        
        # 核心组件
        self.clock = WorldClock()
        self.environment = Environment(
            initial_price=self.config.initial_price,
            base_volatility=self.config.base_volatility,
        )
        self.population = Population(
            max_size=self.config.max_population_size,
            enable_carrying_capacity=self.config.enable_carrying_capacity,
        )
        
        # 遗传机制
        self.gene_pool = GenePool()
        self.mutation_engine = MutationEngine(
            base_mutation_rate=self.config.mutation_rate,
        )
        
        # 生命周期
        self.death_mechanism = DeathMechanism(
            survival_threshold=self.config.survival_threshold,
            enable_natural_death=self.config.enable_natural_death,
            max_lifespan=self.config.max_lifespan,
        )
        self.reproduction_engine = ReproductionEngine(
            mutation_rate=self.config.mutation_rate,
            crossover_rate=self.config.crossover_rate,
            gene_pool=self.gene_pool,
        )
        self.birth_factory = BirthFactory(gene_pool=self.gene_pool)
        
        # 存储
        self.event_store = EventStore(persist_path=self.config.persist_path)
        self.graveyard = Graveyard(persist_path=self.config.persist_path)
        self.snapshot_store = SnapshotStore(
            persist_path=self.config.persist_path,
            auto_snapshot_interval=self.config.snapshot_interval,
        )
        
        # 回调
        self._tick_callbacks: List[Callable[[TickResult], None]] = []
        self._death_callbacks: List[Callable[[DeathCertificate], None]] = []
        self._birth_callbacks: List[Callable[[Agent], None]] = []
        
        # 状态
        self._initialized = False
        self._running = False
    
    def initialize(self) -> None:
        """
        初始化世界
        
        创建初始种群
        """
        if self._initialized:
            return
        
        # 记录模拟开始
        self.event_store.create_event(
            EventType.SIMULATION_STARTED,
            timestamp=self.clock.current_tick,
            payload={"config": self.config.__dict__},
        )
        
        # 创建初始种群
        initial_agents = self.birth_factory.create_population(
            size=self.config.initial_population_size,
            current_tick=self.clock.current_tick,
        )
        
        for agent in initial_agents:
            self.population.add(agent)
            
            # 记录出生事件
            self.event_store.create_event(
                EventType.AGENT_BORN,
                timestamp=self.clock.current_tick,
                agent_id=agent.agent_id,
                payload={
                    "dna_id": agent.dna.dna_id,
                    "generation": agent.lineage.generation,
                    "initial_capital": agent.resources.capital,
                },
            )
        
        # 创建初始快照
        self.snapshot_store.create_snapshot(
            tick=self.clock.current_tick,
            environment=self.environment,
            agents=self.population.get_all(),
        )
        
        self._initialized = True
    
    def tick(self) -> TickResult:
        """
        执行一个世界刻度
        
        这是世界的心跳，所有事件都在这里发生。
        
        顺序：
        1. 环境演化
        2. Agent行动
        3. 订单执行
        4. 死亡检查
        5. 繁殖处理
        6. 状态更新
        
        Returns:
            该tick的结果
        """
        if not self._initialized:
            self.initialize()
        
        current_tick = self.clock.current_tick
        result = TickResult(tick=current_tick, market_state={})
        
        # 1. 环境演化
        market_state = self.environment.tick()
        result.market_state = {
            "price": market_state.price,
            "volatility": market_state.volatility,
            "regime": market_state.regime.name,
        }
        
        # 记录市场事件
        self.event_store.create_event(
            EventType.MARKET_TICK,
            timestamp=current_tick,
            payload=result.market_state,
        )
        
        # 2. 收集所有Agent的行动
        orders: List[Order] = []
        order_to_agent: Dict[str, Agent] = {}
        
        for agent in list(self.population.iter_alive()):
            order = agent.live_one_tick(self.environment, current_tick)
            if order:
                orders.append(order)
                order_to_agent[order.agent_id] = agent
                
                self.event_store.create_event(
                    EventType.ORDER_PLACED,
                    timestamp=current_tick,
                    agent_id=agent.agent_id,
                    payload={
                        "direction": order.direction,
                        "quantity": order.quantity,
                    },
                )
        
        # 3. 执行订单
        if orders:
            execution_results = self.environment.execute(orders)
            
            for result_exec in execution_results:
                if result_exec.executed:
                    result.trades += 1
                    agent = order_to_agent.get(result_exec.order.agent_id)
                    if agent:
                        agent.process_execution(result_exec, current_tick)
                        
                        self.event_store.create_event(
                            EventType.TRADE_EXECUTED,
                            timestamp=current_tick,
                            agent_id=agent.agent_id,
                            payload={
                                "price": result_exec.executed_price,
                                "quantity": result_exec.executed_quantity,
                                "slippage": result_exec.slippage,
                            },
                        )
        
        # 4. 死亡检查
        deaths = []
        for agent in list(self.population.iter_alive()):
            certificate = self.death_mechanism.check_death(
                agent, self.environment, current_tick
            )
            if certificate:
                self.death_mechanism.execute_death(agent, certificate)
                self.population.update_status(agent)
                self.graveyard.bury(certificate)
                deaths.append(certificate)
                
                self.event_store.create_event(
                    EventType.AGENT_DIED,
                    timestamp=current_tick,
                    agent_id=agent.agent_id,
                    payload={
                        "cause": certificate.cause.name,
                        "lifespan": certificate.lifespan,
                        "final_capital": certificate.final_capital,
                        "total_pnl": certificate.total_pnl,
                    },
                )
                
                # 调用死亡回调
                for callback in self._death_callbacks:
                    callback(certificate)
        
        result.deaths = len(deaths)
        
        # 5. 繁殖处理
        births = []
        reproducible = self.population.get_reproducible()
        
        for agent in reproducible:
            # 检查是否达到承载力上限
            if (self.config.enable_carrying_capacity and 
                self.population.alive_count >= self.config.max_population_size):
                break
            
            # 决定繁殖模式
            if self.config.reproduction_mode == "asexual":
                mate = None
            elif self.config.reproduction_mode == "sexual":
                mate = self.reproduction_engine.find_mate(
                    agent, self.population.get_alive()
                )
                if mate is None:
                    continue
            else:  # mixed
                if len(reproducible) > 1 and random.random() < 0.5:
                    mate = self.reproduction_engine.find_mate(
                        agent, self.population.get_alive()
                    )
                else:
                    mate = None
            
            # 尝试繁殖
            repro_result = self.reproduction_engine.attempt_reproduction(
                agent, mate, current_tick
            )
            
            if repro_result.success and repro_result.child_dna:
                # 创建子代
                child = self.birth_factory.create_from_reproduction(
                    repro_result.child_dna,
                    agent,
                    mate,
                    current_tick,
                )
                
                if self.population.add(child):
                    births.append(child)
                    
                    self.event_store.create_event(
                        EventType.AGENT_BORN,
                        timestamp=current_tick,
                        agent_id=child.agent_id,
                        payload={
                            "dna_id": child.dna.dna_id,
                            "generation": child.lineage.generation,
                            "parents": child.lineage.parent_ids,
                            "mutations": [m.name for m in repro_result.mutations_applied],
                        },
                    )
                    
                    self.event_store.create_event(
                        EventType.REPRODUCTION_SUCCEEDED,
                        timestamp=current_tick,
                        agent_id=agent.agent_id,
                        payload={
                            "child_id": child.agent_id,
                            "crossover": repro_result.crossover_type.name if repro_result.crossover_type else None,
                        },
                    )
                    
                    # 调用出生回调
                    for callback in self._birth_callbacks:
                        callback(child)
        
        result.births = len(births)
        
        # 6. 更新统计
        stats = self.population.get_statistics()
        result.alive_count = stats.alive
        result.total_capital = stats.total_capital
        
        # 7. 快照检查
        if self.snapshot_store.should_snapshot(current_tick):
            self.snapshot_store.create_snapshot(
                tick=current_tick,
                environment=self.environment,
                agents=self.population.get_all(),
            )
            
            self.event_store.create_event(
                EventType.SNAPSHOT_CREATED,
                timestamp=current_tick,
            )
        
        # 8. 推进时钟
        self.clock.tick()
        
        # 9. 调用tick回调
        for callback in self._tick_callbacks:
            callback(result)
        
        return result
    
    def run(
        self, 
        ticks: int, 
        progress_callback: Optional[Callable[[int, int], None]] = None,
        stop_condition: Optional[Callable[[TickResult], bool]] = None,
    ) -> List[TickResult]:
        """
        运行模拟
        
        Args:
            ticks: 运行的tick数
            progress_callback: 进度回调 (current, total)
            stop_condition: 停止条件，返回True时停止
            
        Returns:
            所有tick的结果
        """
        self._running = True
        results = []
        
        try:
            for i in range(ticks):
                if not self._running:
                    break
                
                result = self.tick()
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, ticks)
                
                if stop_condition and stop_condition(result):
                    break
                
                # 检查种群是否灭绝
                if result.alive_count == 0:
                    print(f"⚠️ 种群灭绝于 tick {result.tick}")
                    break
        
        finally:
            self._running = False
            
            # 记录模拟结束
            self.event_store.create_event(
                EventType.SIMULATION_ENDED,
                timestamp=self.clock.current_tick,
                payload={
                    "total_ticks": len(results),
                    "final_population": self.population.alive_count,
                    "total_deaths": self.graveyard.__len__(),
                },
            )
        
        return results
    
    def stop(self) -> None:
        """停止模拟"""
        self._running = False
    
    def trigger_extinction_event(self, survival_rate: float = 0.1) -> List[DeathCertificate]:
        """
        触发大灭绝事件
        
        Args:
            survival_rate: 存活率
            
        Returns:
            死亡证明列表
        """
        certificates = self.death_mechanism.trigger_extinction_event(
            self.population.get_alive(),
            survival_rate,
            self.clock.current_tick,
        )
        
        for cert in certificates:
            self.graveyard.bury(cert)
            self.population.update_status(self.population.get(cert.agent_id))
        
        self.event_store.create_event(
            EventType.EXTINCTION_EVENT,
            timestamp=self.clock.current_tick,
            payload={
                "survival_rate": survival_rate,
                "deaths": len(certificates),
            },
        )
        
        return certificates
    
    def register_tick_callback(self, callback: Callable[[TickResult], None]) -> None:
        """注册tick回调"""
        self._tick_callbacks.append(callback)
    
    def register_death_callback(self, callback: Callable[[DeathCertificate], None]) -> None:
        """注册死亡回调"""
        self._death_callbacks.append(callback)
    
    def register_birth_callback(self, callback: Callable[[Agent], None]) -> None:
        """注册出生回调"""
        self._birth_callbacks.append(callback)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取世界摘要"""
        pop_stats = self.population.get_statistics()
        
        return {
            "tick": self.clock.current_tick,
            "population": {
                "alive": pop_stats.alive,
                "dead": pop_stats.dead,
                "total": pop_stats.total,
                "unique_dna": pop_stats.unique_dna,
                "generations": pop_stats.generations,
            },
            "market": self.environment.get_market_summary(),
            "graveyard": {
                "total": len(self.graveyard),
            },
            "events": {
                "total": len(self.event_store),
            },
        }
    
    def __repr__(self) -> str:
        return (
            f"World(tick={self.clock.current_tick}, "
            f"pop={self.population.alive_count}, "
            f"dead={self.population.dead_count})"
        )


# 导入random用于繁殖模式选择
import random
