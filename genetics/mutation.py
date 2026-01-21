"""
突变引擎 - 基因变异的机制

核心原则:
- 突变是随机的，不是定向的
- 突变可能有害、有益或中性
- 大多数突变是中性或有害的
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import random
import math

if TYPE_CHECKING:
    from core.gene import Gene
    from core.dna import DNA


class MutationType(Enum):
    """突变类型"""
    # 参数突变
    POINT_MUTATION = auto()         # 点突变：单个参数微调
    PARAMETER_RESET = auto()        # 参数重置：参数恢复到随机值
    
    # 结构突变
    GENE_DELETION = auto()          # 基因缺失
    GENE_DUPLICATION = auto()       # 基因重复
    GENE_INSERTION = auto()         # 基因插入（从基因库）
    
    # 染色体突变
    CHROMOSOME_INVERSION = auto()   # 染色体倒位
    CHROMOSOME_TRANSLOCATION = auto()  # 染色体易位
    
    # 表达突变
    EXPRESSION_THRESHOLD = auto()   # 表达阈值变化
    DORMANT_ACTIVATION = auto()     # 休眠基因激活
    GENE_SILENCING = auto()         # 基因沉默


@dataclass
class MutationEvent:
    """突变事件记录"""
    mutation_type: MutationType
    gene_id: Optional[str]
    chromosome_name: Optional[str]
    original_value: Any
    new_value: Any
    description: str


class MutationEngine:
    """
    突变引擎
    
    负责对DNA和基因进行各种类型的突变。
    
    关键原则：
    - 突变是随机的，不考虑"好坏"
    - 突变率可配置
    - 记录所有突变事件
    """
    
    def __init__(
        self,
        base_mutation_rate: float = 0.1,
        structural_mutation_rate: float = 0.05,
        enable_gene_pool: bool = True,
    ):
        """
        初始化突变引擎
        
        Args:
            base_mutation_rate: 基础突变率（参数突变）
            structural_mutation_rate: 结构突变率
            enable_gene_pool: 是否启用基因库引入
        """
        self.base_mutation_rate = base_mutation_rate
        self.structural_mutation_rate = structural_mutation_rate
        self.enable_gene_pool = enable_gene_pool
        
        # 突变类型权重
        self._mutation_weights = {
            # 参数突变（最常见）
            MutationType.POINT_MUTATION: 0.5,
            MutationType.PARAMETER_RESET: 0.1,
            
            # 结构突变（较少）
            MutationType.GENE_DELETION: 0.08,
            MutationType.GENE_DUPLICATION: 0.1,
            MutationType.GENE_INSERTION: 0.08,
            
            # 染色体突变（罕见）
            MutationType.CHROMOSOME_INVERSION: 0.03,
            MutationType.CHROMOSOME_TRANSLOCATION: 0.02,
            
            # 表达突变
            MutationType.EXPRESSION_THRESHOLD: 0.05,
            MutationType.DORMANT_ACTIVATION: 0.03,
            MutationType.GENE_SILENCING: 0.01,
        }
        
        # 突变历史
        self._mutation_history: List[MutationEvent] = []
    
    def mutate_gene(
        self, 
        gene: Gene, 
        mutation_rate: Optional[float] = None
    ) -> tuple[Gene, List[MutationEvent]]:
        """
        对单个基因进行突变
        
        Args:
            gene: 要突变的基因
            mutation_rate: 突变率（使用默认值如果不指定）
            
        Returns:
            (突变后的基因, 突变事件列表)
        """
        rate = mutation_rate or self.base_mutation_rate
        events = []
        
        # 决定是否发生突变
        if random.random() >= rate:
            return gene, events
        
        # 选择突变类型
        mutation_type = self._select_mutation_type(is_structural=False)
        
        # 克隆基因
        mutated_gene = gene.clone()
        mutated_gene.mutation_count += 1
        mutated_gene.lineage_depth += 1
        
        if mutation_type == MutationType.POINT_MUTATION:
            event = self._apply_point_mutation(mutated_gene)
        elif mutation_type == MutationType.PARAMETER_RESET:
            event = self._apply_parameter_reset(mutated_gene)
        elif mutation_type == MutationType.EXPRESSION_THRESHOLD:
            event = self._apply_expression_threshold_mutation(mutated_gene)
        else:
            # 对于结构突变，返回原基因（结构突变在DNA层面处理）
            return gene, events
        
        if event:
            events.append(event)
            self._mutation_history.append(event)
        
        return mutated_gene, events
    
    def mutate_dna(
        self, 
        dna: DNA,
        gene_pool: Optional[Any] = None,  # GenePool
    ) -> tuple[DNA, List[MutationEvent]]:
        """
        对DNA进行突变
        
        Args:
            dna: 要突变的DNA
            gene_pool: 基因库（用于基因插入突变）
            
        Returns:
            (突变后的DNA, 突变事件列表)
        """
        events = []
        
        # 对每个染色体进行处理
        for chromosome in dna.chromosomes:
            # 参数突变：对每个基因进行突变检查
            for i, gene in enumerate(chromosome.genes):
                mutated_gene, gene_events = self.mutate_gene(gene)
                if mutated_gene != gene:
                    chromosome.genes[i] = mutated_gene
                    events.extend(gene_events)
            
            # 结构突变检查
            if random.random() < self.structural_mutation_rate:
                structural_events = self._apply_structural_mutation(
                    chromosome, gene_pool
                )
                events.extend(structural_events)
        
        # 染色体级别突变检查
        if random.random() < self.structural_mutation_rate * 0.5:
            chrom_events = self._apply_chromosome_mutation(dna)
            events.extend(chrom_events)
        
        # 重置DNA ID
        dna._dna_id = ""
        
        return dna, events
    
    def _select_mutation_type(self, is_structural: bool = False) -> MutationType:
        """选择突变类型"""
        if is_structural:
            # 只选择结构突变
            structural_types = [
                MutationType.GENE_DELETION,
                MutationType.GENE_DUPLICATION,
                MutationType.GENE_INSERTION,
            ]
            return random.choice(structural_types)
        
        # 按权重选择
        types = list(self._mutation_weights.keys())
        weights = list(self._mutation_weights.values())
        return random.choices(types, weights=weights)[0]
    
    def _apply_point_mutation(self, gene: Gene) -> Optional[MutationEvent]:
        """应用点突变"""
        if not gene.parameters:
            return None
        
        # 随机选择一个参数
        param_name = random.choice(list(gene.parameters.keys()))
        original_value = gene.parameters[param_name]
        
        # 根据参数类型进行突变
        if isinstance(original_value, (int, float)):
            # 数值参数：高斯扰动
            std = abs(original_value) * 0.2 if original_value != 0 else 0.1
            change = random.gauss(0, std)
            new_value = original_value + change
            
            # 保持类型
            if isinstance(original_value, int):
                new_value = int(round(new_value))
            
            gene.parameters[param_name] = new_value
        elif isinstance(original_value, bool):
            # 布尔参数：翻转
            new_value = not original_value
            gene.parameters[param_name] = new_value
        else:
            # 其他类型：不处理
            return None
        
        gene._gene_id = ""  # 重置ID
        
        return MutationEvent(
            mutation_type=MutationType.POINT_MUTATION,
            gene_id=gene.gene_id,
            chromosome_name=None,
            original_value=original_value,
            new_value=new_value,
            description=f"Point mutation on {param_name}: {original_value} -> {new_value}",
        )
    
    def _apply_parameter_reset(self, gene: Gene) -> Optional[MutationEvent]:
        """应用参数重置"""
        if not gene.parameters:
            return None
        
        param_name = random.choice(list(gene.parameters.keys()))
        original_value = gene.parameters[param_name]
        
        # 生成新的随机值
        if isinstance(original_value, (int, float)):
            # 在原值的50%-200%范围内随机
            scale = random.uniform(0.5, 2.0)
            new_value = original_value * scale
            if isinstance(original_value, int):
                new_value = int(round(new_value))
        elif isinstance(original_value, bool):
            new_value = random.choice([True, False])
        else:
            return None
        
        gene.parameters[param_name] = new_value
        gene._gene_id = ""
        
        return MutationEvent(
            mutation_type=MutationType.PARAMETER_RESET,
            gene_id=gene.gene_id,
            chromosome_name=None,
            original_value=original_value,
            new_value=new_value,
            description=f"Parameter reset on {param_name}: {original_value} -> {new_value}",
        )
    
    def _apply_expression_threshold_mutation(self, gene: Gene) -> Optional[MutationEvent]:
        """应用表达阈值突变"""
        original = gene.expression_threshold
        change = random.gauss(0, 0.1)
        new_value = max(0.1, min(0.9, original + change))
        gene.expression_threshold = new_value
        
        return MutationEvent(
            mutation_type=MutationType.EXPRESSION_THRESHOLD,
            gene_id=gene.gene_id,
            chromosome_name=None,
            original_value=original,
            new_value=new_value,
            description=f"Expression threshold: {original:.2f} -> {new_value:.2f}",
        )
    
    def _apply_structural_mutation(
        self, 
        chromosome: Any,  # Chromosome
        gene_pool: Optional[Any] = None,
    ) -> List[MutationEvent]:
        """应用结构突变"""
        events = []
        mutation_type = self._select_mutation_type(is_structural=True)
        
        if mutation_type == MutationType.GENE_DELETION:
            if len(chromosome.genes) > 1:  # 保留至少一个基因
                idx = random.randint(0, len(chromosome.genes) - 1)
                deleted_gene = chromosome.genes.pop(idx)
                events.append(MutationEvent(
                    mutation_type=MutationType.GENE_DELETION,
                    gene_id=deleted_gene.gene_id,
                    chromosome_name=chromosome.name,
                    original_value=deleted_gene.gene_type.name,
                    new_value=None,
                    description=f"Gene deletion: {deleted_gene.gene_type.name}",
                ))
        
        elif mutation_type == MutationType.GENE_DUPLICATION:
            if chromosome.genes:
                idx = random.randint(0, len(chromosome.genes) - 1)
                duplicated = chromosome.genes[idx].clone()
                # 插入到随机位置
                insert_idx = random.randint(0, len(chromosome.genes))
                chromosome.genes.insert(insert_idx, duplicated)
                events.append(MutationEvent(
                    mutation_type=MutationType.GENE_DUPLICATION,
                    gene_id=duplicated.gene_id,
                    chromosome_name=chromosome.name,
                    original_value=None,
                    new_value=duplicated.gene_type.name,
                    description=f"Gene duplication: {duplicated.gene_type.name}",
                ))
        
        elif mutation_type == MutationType.GENE_INSERTION:
            if gene_pool and self.enable_gene_pool:
                # 从基因库获取新基因
                # 根据染色体名称决定基因类型
                from core.gene import GeneType
                type_map = {
                    "signal": GeneType.SIGNAL,
                    "position": GeneType.POSITION,
                    "risk": GeneType.RISK,
                    "dormant": GeneType.DORMANT,
                }
                gene_type = type_map.get(chromosome.name, GeneType.SIGNAL)
                new_gene = gene_pool.get_random_gene(gene_type)
                
                if new_gene:
                    chromosome.genes.append(new_gene)
                    events.append(MutationEvent(
                        mutation_type=MutationType.GENE_INSERTION,
                        gene_id=new_gene.gene_id,
                        chromosome_name=chromosome.name,
                        original_value=None,
                        new_value=new_gene.gene_type.name,
                        description=f"Gene insertion from pool: {new_gene.gene_type.name}",
                    ))
        
        return events
    
    def _apply_chromosome_mutation(self, dna: DNA) -> List[MutationEvent]:
        """应用染色体级别突变"""
        events = []
        
        mutation_type = random.choice([
            MutationType.CHROMOSOME_INVERSION,
            MutationType.CHROMOSOME_TRANSLOCATION,
        ])
        
        if mutation_type == MutationType.CHROMOSOME_INVERSION:
            # 随机选择一个染色体，反转其基因顺序
            if dna.chromosomes:
                chrom = random.choice(dna.chromosomes)
                if len(chrom.genes) > 1:
                    chrom.genes.reverse()
                    events.append(MutationEvent(
                        mutation_type=MutationType.CHROMOSOME_INVERSION,
                        gene_id=None,
                        chromosome_name=chrom.name,
                        original_value="forward",
                        new_value="reversed",
                        description=f"Chromosome inversion: {chrom.name}",
                    ))
        
        elif mutation_type == MutationType.CHROMOSOME_TRANSLOCATION:
            # 在两个染色体之间交换基因
            if len(dna.chromosomes) >= 2:
                chrom1, chrom2 = random.sample(dna.chromosomes, 2)
                if chrom1.genes and chrom2.genes:
                    idx1 = random.randint(0, len(chrom1.genes) - 1)
                    idx2 = random.randint(0, len(chrom2.genes) - 1)
                    
                    # 交换
                    chrom1.genes[idx1], chrom2.genes[idx2] = chrom2.genes[idx2], chrom1.genes[idx1]
                    
                    events.append(MutationEvent(
                        mutation_type=MutationType.CHROMOSOME_TRANSLOCATION,
                        gene_id=None,
                        chromosome_name=f"{chrom1.name}<->{chrom2.name}",
                        original_value=None,
                        new_value=None,
                        description=f"Translocation between {chrom1.name} and {chrom2.name}",
                    ))
        
        return events
    
    def get_mutation_history(self, limit: int = 100) -> List[MutationEvent]:
        """获取突变历史"""
        return self._mutation_history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取突变统计"""
        type_counts = {}
        for event in self._mutation_history:
            type_name = event.mutation_type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        return {
            "total_mutations": len(self._mutation_history),
            "by_type": type_counts,
        }
    
    def __repr__(self) -> str:
        return f"MutationEngine(rate={self.base_mutation_rate}, mutations={len(self._mutation_history)})"
