"""
出生机制 - Agent的创建与初始化

核心原则:
- 新生Agent可以有不同的来源
- 每个新生Agent都是独立的个体
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
import random

from core.agent import Agent, Lineage
from resources import AgentResources
from core.dna import DNA

if TYPE_CHECKING:
    from genetics.gene_pool import GenePool


@dataclass
class BirthConfig:
    """出生配置"""
    initial_capital: float = 10000.0
    capital_variance: float = 0.2  # 资本随机变化幅度
    initial_genes_per_chromosome: int = 2


class BirthFactory:
    """
    出生工厂
    
    负责创建新的Agent，包括：
    1. 从DNA创建（繁殖产生）
    2. 随机创建（自然发生）
    3. 从配置创建（实验用）
    """
    
    def __init__(
        self,
        gene_pool: Optional[GenePool] = None,
        config: Optional[BirthConfig] = None,
    ):
        """
        初始化出生工厂
        
        Args:
            gene_pool: 基因库
            config: 出生配置
        """
        self.gene_pool = gene_pool
        self.config = config or BirthConfig()
    
    def create_from_reproduction(
        self,
        child_dna: DNA,
        parent1: Agent,
        parent2: Optional[Agent],
        current_tick: int,
    ) -> Agent:
        """
        从繁殖创建Agent
        
        Args:
            child_dna: 子代DNA
            parent1: 父代1
            parent2: 父代2（无性繁殖时为None）
            current_tick: 当前时刻
            
        Returns:
            新生Agent
        """
        # 从父代获取资源
        _, child_resources = parent1.prepare_reproduction()
        if parent2:
            _, additional_resources = parent2.prepare_reproduction()
            # 合并资源
            child_resources.capital += additional_resources.capital * 0.5
        
        # 创建血统
        lineage = Lineage.from_parents(parent1, parent2)
        lineage.birth_tick = current_tick
        
        # 创建Agent
        agent = Agent(
            dna=child_dna,
            resources=child_resources,
            lineage=lineage,
            birth_tick=current_tick,
        )
        
        return agent
    
    def create_random(
        self,
        current_tick: int,
        generation: int = 0,
    ) -> Agent:
        """
        随机创建Agent（创世用）
        
        Args:
            current_tick: 当前时刻
            generation: 代数
            
        Returns:
            随机生成的Agent
        """
        # 创建随机DNA
        dna = self._create_random_dna(current_tick, generation)
        
        # 创建随机资源
        resources = self._create_random_resources()
        
        # 创建血统（无父代）
        lineage = Lineage(
            parent_ids=[],
            generation=generation,
            birth_tick=current_tick,
        )
        
        # 创建Agent
        agent = Agent(
            dna=dna,
            resources=resources,
            lineage=lineage,
            birth_tick=current_tick,
        )
        
        return agent
    
    def create_population(
        self,
        size: int,
        current_tick: int,
        diversity: float = 1.0,
    ) -> List[Agent]:
        """
        创建初始种群
        
        Args:
            size: 种群大小
            current_tick: 当前时刻
            diversity: 多样性程度 [0, 1]
            
        Returns:
            Agent列表
        """
        population = []
        
        for i in range(size):
            agent = self.create_random(current_tick, generation=0)
            population.append(agent)
        
        return population
    
    def _create_random_dna(
        self, 
        current_tick: int,
        generation: int
    ) -> DNA:
        """创建随机DNA"""
        from core.gene import (
            SimpleSignalGene, 
            SimplePositionGene, 
            SimpleRiskGene,
            DormantGene,
            GeneType,
        )
        
        dna = DNA()
        dna.creation_tick = current_tick
        dna.generation = generation
        
        # 如果有基因库，从基因库获取
        if self.gene_pool:
            # 信号基因
            for _ in range(random.randint(1, 3)):
                gene = self.gene_pool.get_random_gene(GeneType.SIGNAL)
                if gene:
                    dna.add_gene(gene)
            
            # 仓位基因
            gene = self.gene_pool.get_random_gene(GeneType.POSITION)
            if gene:
                dna.add_gene(gene)
            
            # 风险基因
            gene = self.gene_pool.get_random_gene(GeneType.RISK)
            if gene:
                dna.add_gene(gene)
        else:
            # 使用默认基因
            # 添加信号基因
            signal_gene = SimpleSignalGene(
                parameters={
                    "indicator": random.choice(["price_momentum", "volume", "volatility"]),
                    "threshold_long": random.uniform(0.01, 0.05),
                    "threshold_short": random.uniform(-0.05, -0.01),
                    "lookback": random.randint(5, 50),
                },
                creation_tick=current_tick,
            )
            dna.add_gene(signal_gene)
            
            # 可能添加第二个信号基因
            if random.random() < 0.5:
                signal_gene2 = SimpleSignalGene(
                    parameters={
                        "indicator": "price_momentum",
                        "threshold_long": random.uniform(0.01, 0.05),
                        "threshold_short": random.uniform(-0.05, -0.01),
                        "lookback": random.randint(10, 100),
                    },
                    creation_tick=current_tick,
                )
                dna.add_gene(signal_gene2)
            
            # 添加仓位基因
            position_gene = SimplePositionGene(
                parameters={
                    "base_position": random.uniform(0.05, 0.3),
                    "max_position": random.uniform(0.3, 0.8),
                },
                creation_tick=current_tick,
            )
            dna.add_gene(position_gene)
            
            # 添加风险基因
            risk_gene = SimpleRiskGene(
                parameters={
                    "stop_loss": random.uniform(0.02, 0.1),
                    "risk_tolerance": random.uniform(0.3, 0.8),
                },
                creation_tick=current_tick,
            )
            dna.add_gene(risk_gene)
            
            # 可能添加休眠基因
            if random.random() < 0.3:
                dormant_gene = DormantGene(
                    parameters={
                        "activation_condition": random.choice([
                            "high_volatility", 
                            "crisis", 
                            "drawdown"
                        ]),
                        "activation_threshold": random.uniform(0.03, 0.1),
                    },
                    creation_tick=current_tick,
                )
                dna.add_gene(dormant_gene)
        
        return dna
    
    def _create_random_resources(self) -> AgentResources:
        """创建随机资源"""
        base_capital = self.config.initial_capital
        variance = self.config.capital_variance
        
        # 资本随机变化
        capital = base_capital * (1 + random.uniform(-variance, variance))
        
        # 创建资源
        resources = AgentResources(
            capital=capital,
            metabolism_rate=random.uniform(0.5, 2.0),  # 随机代谢率
        )
        
        return resources
    
    def clone_agent(
        self, 
        agent: Agent, 
        current_tick: int,
        mutate: bool = True,
        mutation_rate: float = 0.1,
    ) -> Agent:
        """
        克隆Agent
        
        Args:
            agent: 要克隆的Agent
            current_tick: 当前时刻
            mutate: 是否应用突变
            mutation_rate: 突变率
            
        Returns:
            克隆的Agent
        """
        # 克隆DNA
        new_dna = agent.dna.clone()
        new_dna.generation = agent.dna.generation + 1
        new_dna.parent_ids = [agent.dna.dna_id]
        new_dna.creation_tick = current_tick
        
        # 应用突变
        if mutate:
            for chrom in new_dna.chromosomes:
                for i, gene in enumerate(chrom.genes):
                    if random.random() < mutation_rate:
                        chrom.genes[i] = gene.mutate(mutation_rate)
        
        # 克隆资源
        new_resources = agent.resources.clone(inheritance_ratio=0.5)
        
        # 创建血统
        lineage = Lineage(
            parent_ids=[agent.agent_id],
            generation=agent.lineage.generation + 1,
            birth_tick=current_tick,
        )
        
        # 创建新Agent
        new_agent = Agent(
            dna=new_dna,
            resources=new_resources,
            lineage=lineage,
            birth_tick=current_tick,
        )
        
        return new_agent
