"""
种群管理 - 管理所有Agent的集合

核心原则:
- 种群是Agent的容器
- 不进行人为筛选
- 只记录和观察
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Set
import random
from collections import Counter
from statistics import median, variance

from core.agent import Agent, AgentState


@dataclass
class PopulationStats:
    """种群统计"""
    total: int = 0
    alive: int = 0
    dead: int = 0
    
    # 代际分布
    generations: Dict[int, int] = field(default_factory=dict)
    
    # DNA多样性
    unique_dna: int = 0
    
    # 资本分布
    total_capital: float = 0.0
    average_capital: float = 0.0
    median_capital: float = 0.0
    mode_capital: float = 0.0
    variance_capital: float = 0.0
    max_capital: float = 0.0
    min_capital: float = float('inf')
    
    # 繁殖统计
    total_offspring: int = 0
    agents_can_reproduce: int = 0


class Population:
    """
    种群
    
    管理所有Agent的容器。
    
    职责：
    1. 存储和索引Agent
    2. 提供各种查询接口
    3. 统计种群状态
    4. 不进行任何人为筛选或干预
    """
    
    def __init__(
        self,
        max_size: Optional[int] = None,
        enable_carrying_capacity: bool = False,
    ):
        """
        初始化种群
        
        Args:
            max_size: 最大种群大小（资源限制）
            enable_carrying_capacity: 是否启用环境承载力限制
        """
        self._agents: Dict[str, Agent] = {}  # agent_id -> Agent
        self._alive_set: Set[str] = set()
        self._dead_set: Set[str] = set()
        
        self.max_size = max_size
        self.enable_carrying_capacity = enable_carrying_capacity
        
        # 索引
        self._generation_index: Dict[int, Set[str]] = {}
        self._dna_index: Dict[str, Set[str]] = {}  # dna_id -> agent_ids
        
        # 统计缓存
        self._stats_cache: Optional[PopulationStats] = None
        self._stats_cache_valid: bool = False
    
    def add(self, agent: Agent) -> bool:
        """
        添加Agent到种群
        
        Args:
            agent: 要添加的Agent
            
        Returns:
            是否成功添加
        """
        # 检查承载力
        if self.enable_carrying_capacity and self.max_size:
            if len(self._alive_set) >= self.max_size:
                return False
        
        # 添加到主存储
        self._agents[agent.agent_id] = agent
        
        # 更新存活/死亡集合
        if agent.is_alive:
            self._alive_set.add(agent.agent_id)
        else:
            self._dead_set.add(agent.agent_id)
        
        # 更新索引
        gen = agent.lineage.generation
        if gen not in self._generation_index:
            self._generation_index[gen] = set()
        self._generation_index[gen].add(agent.agent_id)
        
        dna_id = agent.dna.dna_id
        if dna_id not in self._dna_index:
            self._dna_index[dna_id] = set()
        self._dna_index[dna_id].add(agent.agent_id)
        
        # 使缓存失效
        self._stats_cache_valid = False
        
        return True
    
    def remove(self, agent_id: str) -> Optional[Agent]:
        """
        从种群中移除Agent
        
        注意：通常不应该移除Agent，死亡的Agent应该保留用于研究
        
        Args:
            agent_id: Agent ID
            
        Returns:
            被移除的Agent
        """
        agent = self._agents.pop(agent_id, None)
        if agent:
            self._alive_set.discard(agent_id)
            self._dead_set.discard(agent_id)
            
            # 更新索引
            gen = agent.lineage.generation
            if gen in self._generation_index:
                self._generation_index[gen].discard(agent_id)
            
            dna_id = agent.dna.dna_id
            if dna_id in self._dna_index:
                self._dna_index[dna_id].discard(agent_id)
            
            self._stats_cache_valid = False
        
        return agent
    
    def get(self, agent_id: str) -> Optional[Agent]:
        """获取Agent"""
        return self._agents.get(agent_id)
    
    def update_status(self, agent: Agent) -> None:
        """
        更新Agent状态（存活/死亡）
        
        当Agent死亡时调用
        """
        if agent.agent_id not in self._agents:
            return
        
        if agent.is_alive:
            self._alive_set.add(agent.agent_id)
            self._dead_set.discard(agent.agent_id)
        else:
            self._alive_set.discard(agent.agent_id)
            self._dead_set.add(agent.agent_id)
        
        self._stats_cache_valid = False
    
    def get_alive(self) -> List[Agent]:
        """获取所有存活的Agent"""
        return [self._agents[aid] for aid in self._alive_set]
    
    def get_dead(self) -> List[Agent]:
        """获取所有死亡的Agent"""
        return [self._agents[aid] for aid in self._dead_set]
    
    def get_all(self) -> List[Agent]:
        """获取所有Agent"""
        return list(self._agents.values())
    
    def get_by_generation(self, generation: int) -> List[Agent]:
        """按代获取Agent"""
        agent_ids = self._generation_index.get(generation, set())
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]
    
    def get_by_dna(self, dna_id: str) -> List[Agent]:
        """按DNA获取Agent（携带相同DNA的Agent）"""
        agent_ids = self._dna_index.get(dna_id, set())
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]
    
    def get_reproducible(self) -> List[Agent]:
        """获取可繁殖的Agent"""
        return [a for a in self.get_alive() if a.can_reproduce()]
    
    def sample_alive(self, n: int) -> List[Agent]:
        """随机采样存活Agent"""
        alive = list(self._alive_set)
        if len(alive) <= n:
            return [self._agents[aid] for aid in alive]
        
        sampled_ids = random.sample(alive, n)
        return [self._agents[aid] for aid in sampled_ids]
    
    def iter_alive(self) -> Iterator[Agent]:
        """迭代存活Agent"""
        for agent_id in self._alive_set:
            yield self._agents[agent_id]
    
    def get_statistics(self) -> PopulationStats:
        """获取种群统计"""
        if self._stats_cache_valid and self._stats_cache:
            return self._stats_cache
        
        alive_agents = self.get_alive()
        
        # 代际分布
        generations = {}
        for agent in alive_agents:
            gen = agent.lineage.generation
            generations[gen] = generations.get(gen, 0) + 1
        
        # DNA多样性
        unique_dna = len(set(a.dna.dna_id for a in alive_agents))
        
        # 资本统计
        if alive_agents:
            capitals = [a.resources.capital for a in alive_agents]
            total_capital = sum(capitals)
            average_capital = total_capital / len(capitals)
            max_capital = max(capitals)
            min_capital = min(capitals)
            
            # 中位数
            median_capital = median(capitals)
            
            # 众数（取最频繁出现的资本值，如果有多个则取第一个）
            capital_counts = Counter(capitals)
            if capital_counts:
                mode_capital = capital_counts.most_common(1)[0][0]
            else:
                mode_capital = 0.0
            
            # 方差（需要至少2个数据点）
            if len(capitals) > 1:
                variance_capital = variance(capitals)
            else:
                variance_capital = 0.0
        else:
            total_capital = 0
            average_capital = 0
            median_capital = 0
            mode_capital = 0
            variance_capital = 0
            max_capital = 0
            min_capital = 0
        
        # 繁殖统计
        total_offspring = sum(a.offspring_count for a in self.get_all())
        can_reproduce = len([a for a in alive_agents if a.can_reproduce()])
        
        self._stats_cache = PopulationStats(
            total=len(self._agents),
            alive=len(self._alive_set),
            dead=len(self._dead_set),
            generations=generations,
            unique_dna=unique_dna,
            total_capital=total_capital,
            average_capital=average_capital,
            median_capital=median_capital,
            mode_capital=mode_capital,
            variance_capital=variance_capital,
            max_capital=max_capital,
            min_capital=min_capital,
            total_offspring=total_offspring,
            agents_can_reproduce=can_reproduce,
        )
        self._stats_cache_valid = True
        
        return self._stats_cache
    
    def get_generation_range(self) -> tuple[int, int]:
        """获取代际范围"""
        if not self._generation_index:
            return (0, 0)
        gens = list(self._generation_index.keys())
        return (min(gens), max(gens))
    
    def get_dominant_dna(self, top_n: int = 5) -> List[tuple[str, int]]:
        """
        获取最常见的DNA
        
        Args:
            top_n: 返回前N个
            
        Returns:
            [(dna_id, count), ...]
        """
        # 只统计存活Agent
        dna_counts: Dict[str, int] = {}
        for agent in self.get_alive():
            dna_id = agent.dna.dna_id
            dna_counts[dna_id] = dna_counts.get(dna_id, 0) + 1
        
        sorted_dna = sorted(dna_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_dna[:top_n]
    
    def apply_to_alive(self, func: Callable[[Agent], Any]) -> List[Any]:
        """
        对所有存活Agent应用函数
        
        Args:
            func: 要应用的函数
            
        Returns:
            结果列表
        """
        return [func(agent) for agent in self.iter_alive()]
    
    @property
    def alive_count(self) -> int:
        return len(self._alive_set)
    
    @property
    def dead_count(self) -> int:
        return len(self._dead_set)
    
    def __len__(self) -> int:
        return len(self._agents)
    
    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._agents
    
    def __repr__(self) -> str:
        return f"Population(alive={self.alive_count}, dead={self.dead_count}, total={len(self)})"
