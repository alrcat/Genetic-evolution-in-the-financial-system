"""
基因库 - 可用基因的集合

核心原则:
- 基因库是进化的原材料
- 新基因可以被发现（从基因库引入）
- 基因没有好坏，只有适配环境与否
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type
import random

from core.gene import Gene, GeneType


@dataclass
class GeneTemplate:
    """
    基因模板
    
    定义一种基因的参数范围，用于生成具体基因实例
    """
    gene_class: Type[Gene]
    gene_type: GeneType
    parameter_ranges: Dict[str, tuple]  # 参数名 -> (min, max) 或 可选值列表
    weight: float = 1.0  # 被选中的相对权重
    
    def instantiate(self, creation_tick: int = 0) -> Gene:
        """
        实例化基因
        
        根据参数范围随机生成具体参数
        """
        params = {}
        for name, range_or_choices in self.parameter_ranges.items():
            if isinstance(range_or_choices, (list, tuple)) and len(range_or_choices) == 2:
                # 检查是否为数值范围（排除布尔值）
                if (isinstance(range_or_choices[0], (int, float)) and isinstance(range_or_choices[1], (int, float)) and
                    not isinstance(range_or_choices[0], bool) and not isinstance(range_or_choices[1], bool)):
                    # 数值范围
                    if isinstance(range_or_choices[0], int) and isinstance(range_or_choices[1], int):
                        params[name] = random.randint(range_or_choices[0], range_or_choices[1])
                    else:
                        params[name] = random.uniform(range_or_choices[0], range_or_choices[1])
                else:
                    # 选项列表
                    params[name] = random.choice(range_or_choices)
            elif isinstance(range_or_choices, list):
                # 选项列表
                params[name] = random.choice(range_or_choices)
            else:
                # 固定值
                params[name] = range_or_choices
        
        return self.gene_class(
            parameters=params,
            creation_tick=creation_tick,
        )


class GenePool:
    """
    基因库
    
    管理系统中所有可用的基因模板。
    基因库是进化的原材料来源。
    
    功能：
    1. 存储基因模板
    2. 按类型随机获取基因
    3. 追踪基因使用频率
    4. 支持基因库动态扩展
    """
    
    def __init__(self):
        """初始化基因库"""
        self._templates: Dict[GeneType, List[GeneTemplate]] = {}
        self._usage_count: Dict[str, int] = {}  # gene_class_name -> count
        
        # 初始化默认基因模板
        self._initialize_default_templates()
    
    def _initialize_default_templates(self) -> None:
        """初始化默认基因模板"""
        from core.gene import (
            SimpleSignalGene,
            SimplePositionGene,
            SimpleRiskGene,
            DormantGene,
        )
        
        # 信号基因模板
        self.register_template(GeneTemplate(
            gene_class=SimpleSignalGene,
            gene_type=GeneType.SIGNAL,
            parameter_ranges={
                "indicator": ["price_momentum", "volume_momentum", "volatility_breakout"],
                "threshold_long": (0.005, 0.1),
                "threshold_short": (-0.1, -0.005),
                "lookback": (5, 100),
            },
            weight=2.0,  # 信号基因更常见
        ))
        
        # 趋势跟踪信号
        self.register_template(GeneTemplate(
            gene_class=SimpleSignalGene,
            gene_type=GeneType.SIGNAL,
            parameter_ranges={
                "indicator": ["trend_following"],
                "threshold_long": (0.01, 0.05),
                "threshold_short": (-0.05, -0.01),
                "lookback": (20, 200),
            },
            weight=1.5,
        ))
        
        # 均值回归信号
        self.register_template(GeneTemplate(
            gene_class=SimpleSignalGene,
            gene_type=GeneType.SIGNAL,
            parameter_ranges={
                "indicator": ["mean_reversion"],
                "threshold_long": (-0.05, -0.01),  # 跌多了买
                "threshold_short": (0.01, 0.05),   # 涨多了卖
                "lookback": (10, 50),
            },
            weight=1.0,
        ))
        
        # 仓位基因模板
        self.register_template(GeneTemplate(
            gene_class=SimplePositionGene,
            gene_type=GeneType.POSITION,
            parameter_ranges={
                "base_position": (0.02, 0.5),
                "confidence_scale": [True, False],
                "max_position": (0.3, 1.0),
            },
            weight=1.0,
        ))
        
        # 风险基因模板
        self.register_template(GeneTemplate(
            gene_class=SimpleRiskGene,
            gene_type=GeneType.RISK,
            parameter_ranges={
                "stop_loss": (0.01, 0.15),
                "risk_tolerance": (0.2, 1.0),
                "volatility_sensitivity": (0.5, 2.0),
            },
            weight=1.0,
        ))
        
        # 保守风险基因
        self.register_template(GeneTemplate(
            gene_class=SimpleRiskGene,
            gene_type=GeneType.RISK,
            parameter_ranges={
                "stop_loss": (0.01, 0.03),
                "risk_tolerance": (0.1, 0.3),
                "volatility_sensitivity": (1.5, 3.0),
            },
            weight=0.5,
        ))
        
        # 激进风险基因
        self.register_template(GeneTemplate(
            gene_class=SimpleRiskGene,
            gene_type=GeneType.RISK,
            parameter_ranges={
                "stop_loss": (0.1, 0.2),
                "risk_tolerance": (0.7, 1.0),
                "volatility_sensitivity": (0.2, 0.5),
            },
            weight=0.3,
        ))
        
        # 休眠基因模板 - 高波动触发
        self.register_template(GeneTemplate(
            gene_class=DormantGene,
            gene_type=GeneType.DORMANT,
            parameter_ranges={
                "activation_condition": ["high_volatility"],
                "activation_threshold": (0.03, 0.1),
            },
            weight=0.5,
        ))
        
        # 休眠基因模板 - 危机触发
        self.register_template(GeneTemplate(
            gene_class=DormantGene,
            gene_type=GeneType.DORMANT,
            parameter_ranges={
                "activation_condition": ["crisis"],
                "activation_threshold": (0.05, 0.15),
            },
            weight=0.3,
        ))
        
        # 休眠基因模板 - 回撤触发
        self.register_template(GeneTemplate(
            gene_class=DormantGene,
            gene_type=GeneType.DORMANT,
            parameter_ranges={
                "activation_condition": ["drawdown"],
                "activation_threshold": (0.05, 0.2),
            },
            weight=0.4,
        ))
    
    def register_template(self, template: GeneTemplate) -> None:
        """
        注册基因模板
        
        Args:
            template: 基因模板
        """
        if template.gene_type not in self._templates:
            self._templates[template.gene_type] = []
        self._templates[template.gene_type].append(template)
    
    def get_random_gene(
        self, 
        gene_type: GeneType,
        creation_tick: int = 0,
    ) -> Optional[Gene]:
        """
        随机获取指定类型的基因
        
        Args:
            gene_type: 基因类型
            creation_tick: 创建时刻
            
        Returns:
            随机生成的基因，如果没有该类型模板则返回None
        """
        templates = self._templates.get(gene_type, [])
        if not templates:
            return None
        
        # 按权重选择模板
        weights = [t.weight for t in templates]
        template = random.choices(templates, weights=weights)[0]
        
        # 实例化基因
        gene = template.instantiate(creation_tick)
        
        # 记录使用
        class_name = template.gene_class.__name__
        self._usage_count[class_name] = self._usage_count.get(class_name, 0) + 1
        
        return gene
    
    def get_all_types(self) -> List[GeneType]:
        """获取所有可用的基因类型"""
        return list(self._templates.keys())
    
    def get_template_count(self, gene_type: GeneType) -> int:
        """获取指定类型的模板数量"""
        return len(self._templates.get(gene_type, []))
    
    def get_usage_statistics(self) -> Dict[str, int]:
        """获取基因使用统计"""
        return self._usage_count.copy()
    
    def create_random_gene_set(
        self,
        signal_count: int = 1,
        position_count: int = 1,
        risk_count: int = 1,
        dormant_probability: float = 0.3,
        creation_tick: int = 0,
    ) -> List[Gene]:
        """
        创建随机基因组合
        
        Args:
            signal_count: 信号基因数量
            position_count: 仓位基因数量
            risk_count: 风险基因数量
            dormant_probability: 包含休眠基因的概率
            creation_tick: 创建时刻
            
        Returns:
            基因列表
        """
        genes = []
        
        # 信号基因
        for _ in range(signal_count):
            gene = self.get_random_gene(GeneType.SIGNAL, creation_tick)
            if gene:
                genes.append(gene)
        
        # 仓位基因
        for _ in range(position_count):
            gene = self.get_random_gene(GeneType.POSITION, creation_tick)
            if gene:
                genes.append(gene)
        
        # 风险基因
        for _ in range(risk_count):
            gene = self.get_random_gene(GeneType.RISK, creation_tick)
            if gene:
                genes.append(gene)
        
        # 休眠基因
        if random.random() < dormant_probability:
            gene = self.get_random_gene(GeneType.DORMANT, creation_tick)
            if gene:
                genes.append(gene)
        
        return genes
    
    def __repr__(self) -> str:
        type_counts = {t.name: len(templates) for t, templates in self._templates.items()}
        return f"GenePool({type_counts})"
