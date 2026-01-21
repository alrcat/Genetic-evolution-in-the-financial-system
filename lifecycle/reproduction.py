"""
繁殖机制 - 遗传信息的传递与变异

核心原则:
- 繁殖不是奖励，而是生物本能
- 繁殖不保证后代更优
- 变异是随机的，不是定向的
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple, TYPE_CHECKING
import random
import copy

if TYPE_CHECKING:
    from core.agent import Agent
    from core.dna import DNA, Chromosome
    from core.gene import Gene
    from genetics.gene_pool import GenePool


class CrossoverType(Enum):
    """交叉类型"""
    SINGLE_POINT = auto()   # 单点交叉
    TWO_POINT = auto()      # 两点交叉
    UNIFORM = auto()        # 均匀交叉
    CHROMOSOME = auto()     # 染色体级别交叉


class MutationType(Enum):
    """突变类型"""
    POINT = auto()          # 点突变（参数微调）
    GENE_DELETION = auto()  # 基因缺失
    GENE_DUPLICATION = auto()  # 基因重复
    GENE_INSERTION = auto()    # 基因插入
    CHROMOSOME_INVERSION = auto()  # 染色体倒位
    DORMANT_ACTIVATION = auto()    # 休眠基因激活


@dataclass
class ReproductionResult:
    """繁殖结果"""
    success: bool
    child_dna: Optional[DNA] = None
    mutations_applied: List[MutationType] = None
    crossover_type: Optional[CrossoverType] = None
    error_message: str = ""
    
    def __post_init__(self):
        if self.mutations_applied is None:
            self.mutations_applied = []


class ReproductionEngine:
    """
    繁殖引擎
    
    负责处理Agent的繁殖过程，包括：
    1. 无性繁殖（复制 + 突变）
    2. 有性繁殖（交叉 + 突变）
    
    繁殖不保证后代更优！
    这不是优化算法，而是进化模拟。
    """
    
    def __init__(
        self,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        gene_pool: Optional[GenePool] = None,
    ):
        """
        初始化繁殖引擎
        
        Args:
            mutation_rate: 基础突变率
            crossover_rate: 交叉概率
            gene_pool: 基因库（用于引入新基因）
        """
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.gene_pool = gene_pool
        
        # 各种突变的相对概率
        self.mutation_weights = {
            MutationType.POINT: 0.5,
            MutationType.GENE_DELETION: 0.1,
            MutationType.GENE_DUPLICATION: 0.15,
            MutationType.GENE_INSERTION: 0.15,
            MutationType.CHROMOSOME_INVERSION: 0.05,
            MutationType.DORMANT_ACTIVATION: 0.05,
        }
    
    def attempt_reproduction(
        self,
        parent1: Agent,
        parent2: Optional[Agent] = None,
        current_tick: int = 0,
    ) -> ReproductionResult:
        """
        尝试繁殖
        
        Args:
            parent1: 第一个父代
            parent2: 第二个父代（无性繁殖时为None）
            current_tick: 当前时刻
            
        Returns:
            繁殖结果
        """
        # 检查繁殖资格
        if not parent1.can_reproduce():
            return ReproductionResult(
                success=False,
                error_message="Parent1 cannot reproduce"
            )
        
        if parent2 is not None and not parent2.can_reproduce():
            return ReproductionResult(
                success=False,
                error_message="Parent2 cannot reproduce"
            )
        
        # 执行繁殖
        if parent2 is None:
            return self._asexual_reproduction(parent1, current_tick)
        else:
            return self._sexual_reproduction(parent1, parent2, current_tick)
    
    def _asexual_reproduction(
        self, 
        parent: Agent,
        current_tick: int
    ) -> ReproductionResult:
        """
        无性繁殖
        
        复制父代DNA + 应用突变
        """
        from core.dna import DNA
        
        # 复制DNA
        child_dna = parent.dna.clone()
        child_dna.generation = parent.dna.generation + 1
        child_dna.parent_ids = [parent.dna.dna_id]
        child_dna.creation_tick = current_tick
        
        # 应用突变
        mutations = self._apply_mutations(child_dna)
        
        return ReproductionResult(
            success=True,
            child_dna=child_dna,
            mutations_applied=mutations,
            crossover_type=None,
        )
    
    def _sexual_reproduction(
        self,
        parent1: Agent,
        parent2: Agent,
        current_tick: int
    ) -> ReproductionResult:
        """
        有性繁殖
        
        交叉重组 + 突变
        """
        from core.dna import DNA
        
        # 选择交叉类型
        crossover_type = random.choice([
            CrossoverType.SINGLE_POINT,
            CrossoverType.TWO_POINT,
            CrossoverType.UNIFORM,
            CrossoverType.CHROMOSOME,
        ])
        
        # 执行交叉
        child_dna = self._crossover(
            parent1.dna, 
            parent2.dna, 
            crossover_type
        )
        
        child_dna.generation = max(parent1.dna.generation, parent2.dna.generation) + 1
        child_dna.parent_ids = [parent1.dna.dna_id, parent2.dna.dna_id]
        child_dna.creation_tick = current_tick
        
        # 应用突变
        mutations = self._apply_mutations(child_dna)
        
        return ReproductionResult(
            success=True,
            child_dna=child_dna,
            mutations_applied=mutations,
            crossover_type=crossover_type,
        )
    
    def _crossover(
        self, 
        dna1: DNA, 
        dna2: DNA, 
        crossover_type: CrossoverType
    ) -> DNA:
        """
        执行交叉
        
        Args:
            dna1: 父代1的DNA
            dna2: 父代2的DNA
            crossover_type: 交叉类型
            
        Returns:
            子代DNA
        """
        from core.dna import DNA, Chromosome
        
        if crossover_type == CrossoverType.CHROMOSOME:
            return self._chromosome_crossover(dna1, dna2)
        elif crossover_type == CrossoverType.UNIFORM:
            return self._uniform_crossover(dna1, dna2)
        elif crossover_type == CrossoverType.TWO_POINT:
            return self._two_point_crossover(dna1, dna2)
        else:  # SINGLE_POINT
            return self._single_point_crossover(dna1, dna2)
    
    def _single_point_crossover(self, dna1: DNA, dna2: DNA) -> DNA:
        """单点交叉"""
        from core.dna import DNA, Chromosome
        
        child_chromosomes = []
        
        for i, (c1, c2) in enumerate(zip(dna1.chromosomes, dna2.chromosomes)):
            # 在染色体内选择交叉点
            if len(c1.genes) == 0 or len(c2.genes) == 0:
                # 空染色体，直接复制
                child_chromosomes.append(c1.clone() if random.random() < 0.5 else c2.clone())
                continue
            
            crossover_point = random.randint(0, min(len(c1.genes), len(c2.genes)))
            
            new_genes = []
            new_genes.extend([g.clone() for g in c1.genes[:crossover_point]])
            new_genes.extend([g.clone() for g in c2.genes[crossover_point:]])
            
            child_chromosomes.append(Chromosome(
                name=c1.name,
                genes=new_genes,
                dominance=(c1.dominance + c2.dominance) / 2,
            ))
        
        return DNA(chromosomes=child_chromosomes)
    
    def _two_point_crossover(self, dna1: DNA, dna2: DNA) -> DNA:
        """两点交叉"""
        from core.dna import DNA, Chromosome
        
        child_chromosomes = []
        
        for c1, c2 in zip(dna1.chromosomes, dna2.chromosomes):
            if len(c1.genes) < 2 or len(c2.genes) < 2:
                child_chromosomes.append(c1.clone() if random.random() < 0.5 else c2.clone())
                continue
            
            max_len = min(len(c1.genes), len(c2.genes))
            p1, p2 = sorted(random.sample(range(max_len + 1), 2))
            
            new_genes = []
            new_genes.extend([g.clone() for g in c1.genes[:p1]])
            new_genes.extend([g.clone() for g in c2.genes[p1:p2]])
            new_genes.extend([g.clone() for g in c1.genes[p2:]])
            
            child_chromosomes.append(Chromosome(
                name=c1.name,
                genes=new_genes,
                dominance=(c1.dominance + c2.dominance) / 2,
            ))
        
        return DNA(chromosomes=child_chromosomes)
    
    def _uniform_crossover(self, dna1: DNA, dna2: DNA) -> DNA:
        """均匀交叉"""
        from core.dna import DNA, Chromosome
        
        child_chromosomes = []
        
        for c1, c2 in zip(dna1.chromosomes, dna2.chromosomes):
            new_genes = []
            
            # 对每个位置随机选择来自哪个父代
            max_len = max(len(c1.genes), len(c2.genes))
            for i in range(max_len):
                if random.random() < 0.5:
                    if i < len(c1.genes):
                        new_genes.append(c1.genes[i].clone())
                else:
                    if i < len(c2.genes):
                        new_genes.append(c2.genes[i].clone())
            
            child_chromosomes.append(Chromosome(
                name=c1.name,
                genes=new_genes,
                dominance=(c1.dominance + c2.dominance) / 2,
            ))
        
        return DNA(chromosomes=child_chromosomes)
    
    def _chromosome_crossover(self, dna1: DNA, dna2: DNA) -> DNA:
        """染色体级别交叉"""
        from core.dna import DNA
        
        child_chromosomes = []
        
        for c1, c2 in zip(dna1.chromosomes, dna2.chromosomes):
            # 随机选择整个染色体
            if random.random() < 0.5:
                child_chromosomes.append(c1.clone())
            else:
                child_chromosomes.append(c2.clone())
        
        return DNA(chromosomes=child_chromosomes)
    
    def _apply_mutations(self, dna: DNA) -> List[MutationType]:
        """
        应用突变
        
        Args:
            dna: 要突变的DNA
            
        Returns:
            应用的突变类型列表
        """
        applied_mutations = []
        
        # 对每个基因检查是否突变
        for chromosome in dna.chromosomes:
            genes_to_remove = []
            genes_to_add = []
            
            for i, gene in enumerate(chromosome.genes):
                if random.random() < self.mutation_rate:
                    # 选择突变类型
                    mutation_type = self._select_mutation_type()
                    
                    if mutation_type == MutationType.POINT:
                        # 点突变
                        chromosome.genes[i] = gene.mutate(self.mutation_rate)
                        applied_mutations.append(MutationType.POINT)
                    
                    elif mutation_type == MutationType.GENE_DELETION:
                        # 基因缺失
                        if len(chromosome.genes) > 1:  # 保留至少一个基因
                            genes_to_remove.append(i)
                            applied_mutations.append(MutationType.GENE_DELETION)
                    
                    elif mutation_type == MutationType.GENE_DUPLICATION:
                        # 基因重复
                        genes_to_add.append(gene.clone())
                        applied_mutations.append(MutationType.GENE_DUPLICATION)
                    
                    elif mutation_type == MutationType.GENE_INSERTION:
                        # 基因插入（从基因库）
                        if self.gene_pool is not None:
                            new_gene = self.gene_pool.get_random_gene(gene.gene_type)
                            if new_gene:
                                genes_to_add.append(new_gene)
                                applied_mutations.append(MutationType.GENE_INSERTION)
            
            # 执行删除（倒序）
            for i in reversed(genes_to_remove):
                chromosome.genes.pop(i)
            
            # 执行添加
            chromosome.genes.extend(genes_to_add)
        
        # 重置DNA ID
        dna._dna_id = ""
        
        return applied_mutations
    
    def _select_mutation_type(self) -> MutationType:
        """根据权重选择突变类型"""
        types = list(self.mutation_weights.keys())
        weights = list(self.mutation_weights.values())
        return random.choices(types, weights=weights)[0]
    
    def find_mate(
        self, 
        agent: Agent, 
        population: List[Agent],
        strategy: str = "random"
    ) -> Optional[Agent]:
        """
        寻找配偶
        
        Args:
            agent: 寻找配偶的Agent
            population: 种群
            strategy: 配对策略
            
        Returns:
            配偶（如果找到）
        """
        # 筛选可繁殖的候选者
        candidates = [
            a for a in population 
            if a.agent_id != agent.agent_id 
            and a.can_reproduce()
        ]
        
        if not candidates:
            return None
        
        if strategy == "random":
            return random.choice(candidates)
        
        elif strategy == "similar":
            # 选择DNA相似的
            # 简化实现：选择同代或相近代的
            same_gen = [c for c in candidates 
                       if abs(c.lineage.generation - agent.lineage.generation) <= 2]
            return random.choice(same_gen) if same_gen else random.choice(candidates)
        
        elif strategy == "diverse":
            # 选择DNA差异大的
            # 简化实现：选择不同代的
            diff_gen = [c for c in candidates 
                       if abs(c.lineage.generation - agent.lineage.generation) > 2]
            return random.choice(diff_gen) if diff_gen else random.choice(candidates)
        
        return random.choice(candidates)
