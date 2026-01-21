"""
DNA系统 - 基因组合体

核心原则:
- DNA定义的是可能性，不是必然性
- DNA + 环境 → 表型（实际行为）
- 不同市场环境会激活不同的基因组合
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import hashlib
import json
import random
import copy

from core.gene import Gene, GeneType, GeneExpression, SignalDirection

if TYPE_CHECKING:
    from core.environment import EnvironmentState


@dataclass
class Chromosome:
    """
    染色体 - 一组相关基因的集合
    
    染色体内的基因通常协同工作，例如：
    - 信号染色体：包含多个信号基因
    - 风险染色体：包含风险管理相关基因
    """
    
    name: str
    genes: List[Gene] = field(default_factory=list)
    dominance: float = 1.0  # 显性程度 [0, 1]
    
    # 连锁关系：某些基因倾向于一起遗传
    linkage_groups: List[List[int]] = field(default_factory=list)
    
    def add_gene(self, gene: Gene) -> None:
        """添加基因"""
        self.genes.append(gene)
    
    def remove_gene(self, gene_id: str) -> bool:
        """移除基因"""
        for i, gene in enumerate(self.genes):
            if gene.gene_id == gene_id:
                self.genes.pop(i)
                return True
        return False
    
    def get_genes_by_type(self, gene_type: GeneType) -> List[Gene]:
        """按类型获取基因"""
        return [g for g in self.genes if g.gene_type == gene_type]
    
    def clone(self) -> Chromosome:
        """深拷贝"""
        return Chromosome(
            name=self.name,
            genes=[g.clone() for g in self.genes],
            dominance=self.dominance,
            linkage_groups=copy.deepcopy(self.linkage_groups),
        )
    
    def __len__(self) -> int:
        return len(self.genes)
    
    def __repr__(self) -> str:
        return f"Chromosome({self.name}, genes={len(self.genes)}, dom={self.dominance:.2f})"


@dataclass
class TradeAction:
    """交易行动"""
    direction: SignalDirection
    position_size: float  # 仓位比例
    stop_loss: float      # 止损比例
    confidence: float     # 综合置信度
    energy_cost: float    # 总能量消耗
    
    def __repr__(self) -> str:
        return f"TradeAction({self.direction.name}, size={self.position_size:.2%}, SL={self.stop_loss:.2%})"


@dataclass
class Phenotype:
    """
    表型 - DNA表达后的实际行为
    
    表型是基因与环境相互作用的结果，
    同样的DNA在不同环境下可能产生不同的表型。
    """
    
    expressions: List[GeneExpression] = field(default_factory=list)
    
    # 综合信号
    signal: SignalDirection = SignalDirection.NEUTRAL
    signal_confidence: float = 0.0
    
    # 建议行动
    suggested_position: float = 0.0
    suggested_stop_loss: float = 0.05
    
    # 能量消耗
    total_energy_cost: float = 0.0
    
    # 表达的基因统计
    expressed_genes: int = 0
    dormant_activated: int = 0
    
    def to_action(self) -> Optional[TradeAction]:
        """
        将表型转换为交易行动
        
        Returns:
            交易行动，如果没有明确信号则返回None
        """
        if self.signal == SignalDirection.NEUTRAL or self.signal_confidence < 0.3:
            return None
        
        return TradeAction(
            direction=self.signal,
            position_size=self.suggested_position,
            stop_loss=self.suggested_stop_loss,
            confidence=self.signal_confidence,
            energy_cost=self.total_energy_cost,
        )
    
    def __repr__(self) -> str:
        return (
            f"Phenotype(signal={self.signal.name}, conf={self.signal_confidence:.2f}, "
            f"pos={self.suggested_position:.2%}, expressed={self.expressed_genes})"
        )


@dataclass
class DNA:
    """
    DNA - 基因组合体
    
    DNA定义了Agent的【潜在行为空间】，
    但实际行为（表型）取决于环境激活了哪些基因。
    
    注意：DNA不等于策略！
    DNA定义的是可能性，不是必然性。
    """
    
    chromosomes: List[Chromosome] = field(default_factory=list)
    
    # DNA元数据
    generation: int = 0           # 第几代
    parent_ids: List[str] = field(default_factory=list)  # 父代DNA ID
    creation_tick: int = 0        # 创建时刻
    
    _dna_id: str = field(default="", init=False)
    
    def __post_init__(self):
        """初始化后计算DNA ID"""
        if not self.chromosomes:
            self._initialize_default_chromosomes()
        self._dna_id = self._compute_dna_id()
    
    def _initialize_default_chromosomes(self) -> None:
        """初始化默认染色体结构"""
        self.chromosomes = [
            Chromosome(name="signal", genes=[], dominance=1.0),
            Chromosome(name="position", genes=[], dominance=1.0),
            Chromosome(name="risk", genes=[], dominance=1.0),
            Chromosome(name="dormant", genes=[], dominance=0.5),
        ]
    
    @property
    def dna_id(self) -> str:
        """DNA的唯一标识"""
        if not self._dna_id:
            self._dna_id = self._compute_dna_id()
        return self._dna_id
    
    def _compute_dna_id(self) -> str:
        """计算DNA ID（基于所有基因内容）"""
        all_gene_ids = []
        for chrom in self.chromosomes:
            for gene in chrom.genes:
                all_gene_ids.append(gene.gene_id)
        
        content = {
            "genes": sorted(all_gene_ids),
            "generation": self.generation,
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    def add_gene(self, gene: Gene, chromosome_name: str = None) -> None:
        """
        添加基因到DNA
        
        Args:
            gene: 要添加的基因
            chromosome_name: 目标染色体名称，默认根据基因类型自动选择
        """
        if chromosome_name is None:
            # 根据基因类型选择染色体
            type_to_chrom = {
                GeneType.SIGNAL: "signal",
                GeneType.FILTER: "signal",
                GeneType.POSITION: "position",
                GeneType.EXIT: "position",
                GeneType.RISK: "risk",
                GeneType.TIMING: "signal",
                GeneType.DORMANT: "dormant",
            }
            chromosome_name = type_to_chrom.get(gene.gene_type, "signal")
        
        for chrom in self.chromosomes:
            if chrom.name == chromosome_name:
                chrom.add_gene(gene)
                self._dna_id = ""  # 需要重新计算
                return
        
        # 如果染色体不存在，创建新的
        new_chrom = Chromosome(name=chromosome_name, genes=[gene])
        self.chromosomes.append(new_chrom)
        self._dna_id = ""
    
    def get_all_genes(self) -> List[Gene]:
        """获取所有基因"""
        genes = []
        for chrom in self.chromosomes:
            genes.extend(chrom.genes)
        return genes
    
    def get_genes_by_type(self, gene_type: GeneType) -> List[Gene]:
        """按类型获取基因"""
        genes = []
        for chrom in self.chromosomes:
            genes.extend(chrom.get_genes_by_type(gene_type))
        return genes
    
    def express(self, env_state: EnvironmentState) -> Phenotype:
        """
        基因表达过程：DNA + 环境 → 表型
        
        这不是简单的读取，而是一个复杂的表达过程：
        1. 环境条件筛选：哪些基因被激活
        2. 基因间交互：协同、抑制、竞争
        3. 综合决策：多个基因输出的整合
        
        Args:
            env_state: 当前环境状态
            
        Returns:
            表型（实际行为）
        """
        phenotype = Phenotype()
        
        # 收集所有基因表达
        signal_expressions: List[GeneExpression] = []
        position_expressions: List[GeneExpression] = []
        risk_expressions: List[GeneExpression] = []
        
        for chrom in self.chromosomes:
            for gene in chrom.genes:
                if gene.can_express(env_state):
                    expr = gene.express(env_state)
                    phenotype.expressions.append(expr)
                    
                    if expr.expressed:
                        phenotype.expressed_genes += 1
                        phenotype.total_energy_cost += expr.energy_cost
                        
                        # 分类表达结果
                        if gene.gene_type in [GeneType.SIGNAL, GeneType.FILTER]:
                            signal_expressions.append(expr)
                        elif gene.gene_type == GeneType.POSITION:
                            position_expressions.append(expr)
                        elif gene.gene_type == GeneType.RISK:
                            risk_expressions.append(expr)
                        elif gene.gene_type == GeneType.DORMANT:
                            phenotype.dormant_activated += 1
        
        # 整合信号
        phenotype.signal, phenotype.signal_confidence = self._integrate_signals(signal_expressions)
        
        # 整合仓位建议
        phenotype.suggested_position = self._integrate_positions(position_expressions)
        
        # 整合风险参数
        phenotype.suggested_stop_loss = self._integrate_risk(risk_expressions)
        
        return phenotype
    
    def _integrate_signals(
        self, 
        expressions: List[GeneExpression]
    ) -> Tuple[SignalDirection, float]:
        """
        整合多个信号基因的输出
        
        使用加权投票机制
        """
        if not expressions:
            return SignalDirection.NEUTRAL, 0.0
        
        # 计算加权投票
        long_score = 0.0
        short_score = 0.0
        total_confidence = 0.0
        
        for expr in expressions:
            if expr.output == SignalDirection.LONG:
                long_score += expr.confidence
            elif expr.output == SignalDirection.SHORT:
                short_score += expr.confidence
            total_confidence += expr.confidence
        
        if total_confidence == 0:
            return SignalDirection.NEUTRAL, 0.0
        
        # 决定最终信号
        if long_score > short_score and long_score > 0.5 * total_confidence:
            return SignalDirection.LONG, long_score / total_confidence
        elif short_score > long_score and short_score > 0.5 * total_confidence:
            return SignalDirection.SHORT, short_score / total_confidence
        else:
            return SignalDirection.NEUTRAL, 0.0
    
    def _integrate_positions(self, expressions: List[GeneExpression]) -> float:
        """整合仓位建议"""
        if not expressions:
            return 0.1  # 默认10%
        
        # 取平均
        positions = [e.output for e in expressions if isinstance(e.output, (int, float))]
        if not positions:
            return 0.1
        
        return sum(positions) / len(positions)
    
    def _integrate_risk(self, expressions: List[GeneExpression]) -> float:
        """整合风险参数"""
        if not expressions:
            return 0.05  # 默认5%止损
        
        # 取最保守的止损
        stop_losses = []
        for expr in expressions:
            if isinstance(expr.output, dict) and "stop_loss" in expr.output:
                stop_losses.append(expr.output["stop_loss"])
        
        if not stop_losses:
            return 0.05
        
        return min(stop_losses)  # 最严格的止损
    
    def clone(self) -> DNA:
        """深拷贝DNA"""
        new_dna = DNA(
            chromosomes=[c.clone() for c in self.chromosomes],
            generation=self.generation,
            parent_ids=self.parent_ids.copy(),
            creation_tick=self.creation_tick,
        )
        new_dna._dna_id = ""
        return new_dna
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "dna_id": self.dna_id,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "creation_tick": self.creation_tick,
            "chromosomes": [
                {
                    "name": c.name,
                    "dominance": c.dominance,
                    "genes": [g.to_dict() for g in c.genes],
                }
                for c in self.chromosomes
            ],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DNA:
        """从字典反序列化"""
        # 需要基因工厂来重建具体的基因类型
        # 这里先返回简化版本
        dna = cls(
            generation=data.get("generation", 0),
            parent_ids=data.get("parent_ids", []),
            creation_tick=data.get("creation_tick", 0),
        )
        # TODO: 重建chromosomes和genes
        return dna
    
    def get_complexity(self) -> int:
        """获取DNA复杂度（总基因数）"""
        return sum(len(c) for c in self.chromosomes)
    
    def get_signature(self) -> str:
        """获取DNA签名（简短表示）"""
        gene_types = {}
        for gene in self.get_all_genes():
            gene_types[gene.gene_type.name] = gene_types.get(gene.gene_type.name, 0) + 1
        
        sig_parts = [f"{k[0]}:{v}" for k, v in sorted(gene_types.items())]
        return f"[{','.join(sig_parts)}]"
    
    def __repr__(self) -> str:
        return f"DNA(id={self.dna_id[:8]}, gen={self.generation}, {self.get_signature()})"
    
    def __hash__(self) -> int:
        return hash(self.dna_id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DNA):
            return False
        return self.dna_id == other.dna_id
