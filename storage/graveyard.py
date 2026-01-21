"""
死亡档案 - 记录所有死亡Agent的信息

核心原则:
- 死亡是进化的一部分
- 死亡档案是研究进化历史的重要资源
- 死亡的DNA可能在未来被研究
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import os

from lifecycle.death import DeathCertificate
from core.agent import DeathCause


@dataclass
class GraveyardEntry:
    """墓地条目"""
    certificate: DeathCertificate
    archived_at: datetime
    
    # 额外分析数据
    survival_percentile: float = 0.0  # 在同代中的存活百分位
    pnl_percentile: float = 0.0       # 在同代中的盈亏百分位
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "certificate": {
                "agent_id": self.certificate.agent_id,
                "death_tick": self.certificate.death_tick,
                "death_time": self.certificate.death_time.isoformat(),
                "cause": self.certificate.cause.name,
                "final_capital": self.certificate.final_capital,
                "lifespan": self.certificate.lifespan,
                "generation": self.certificate.generation,
                "offspring_count": self.certificate.offspring_count,
                "total_trades": self.certificate.total_trades,
                "winning_trades": self.certificate.winning_trades,
                "total_pnl": self.certificate.total_pnl,
                "max_drawdown": self.certificate.max_drawdown,
                "dna_archive": self.certificate.dna_archive,
                "parent_ids": self.certificate.parent_ids,
            },
            "archived_at": self.archived_at.isoformat(),
            "survival_percentile": self.survival_percentile,
            "pnl_percentile": self.pnl_percentile,
        }


class Graveyard:
    """
    死亡档案馆
    
    存储所有死亡Agent的信息，用于：
    1. 进化历史研究
    2. 死因分析
    3. DNA谱系追踪
    4. 生存分析
    """
    
    def __init__(self, persist_path: Optional[str] = None):
        """
        初始化死亡档案馆
        
        Args:
            persist_path: 持久化路径（可选）
        """
        self._entries: List[GraveyardEntry] = []
        self._agent_index: Dict[str, int] = {}  # agent_id -> index
        self._cause_index: Dict[DeathCause, List[int]] = {}
        self._generation_index: Dict[int, List[int]] = {}
        
        self._persist_path = persist_path
        
        # 统计缓存
        self._stats_cache: Optional[Dict[str, Any]] = None
        self._stats_cache_valid: bool = False
    
    def bury(self, certificate: DeathCertificate) -> GraveyardEntry:
        """
        安葬Agent
        
        Args:
            certificate: 死亡证明
            
        Returns:
            墓地条目
        """
        entry = GraveyardEntry(
            certificate=certificate,
            archived_at=datetime.now(),
        )
        
        index = len(self._entries)
        self._entries.append(entry)
        
        # 更新索引
        self._agent_index[certificate.agent_id] = index
        
        if certificate.cause not in self._cause_index:
            self._cause_index[certificate.cause] = []
        self._cause_index[certificate.cause].append(index)
        
        if certificate.generation not in self._generation_index:
            self._generation_index[certificate.generation] = []
        self._generation_index[certificate.generation].append(index)
        
        # 使统计缓存失效
        self._stats_cache_valid = False
        
        return entry
    
    def get_by_agent(self, agent_id: str) -> Optional[GraveyardEntry]:
        """按Agent ID获取条目"""
        index = self._agent_index.get(agent_id)
        if index is not None:
            return self._entries[index]
        return None
    
    def get_by_cause(self, cause: DeathCause) -> List[GraveyardEntry]:
        """按死因获取条目"""
        indices = self._cause_index.get(cause, [])
        return [self._entries[i] for i in indices]
    
    def get_by_generation(self, generation: int) -> List[GraveyardEntry]:
        """按代数获取条目"""
        indices = self._generation_index.get(generation, [])
        return [self._entries[i] for i in indices]
    
    def get_by_time_range(
        self, 
        start_tick: int, 
        end_tick: int
    ) -> List[GraveyardEntry]:
        """按死亡时间范围获取条目"""
        return [
            e for e in self._entries
            if start_tick <= e.certificate.death_tick < end_tick
        ]
    
    def get_lineage(self, agent_id: str) -> List[GraveyardEntry]:
        """
        获取Agent的祖先链
        
        Args:
            agent_id: Agent ID
            
        Returns:
            祖先条目列表（从最近到最远）
        """
        lineage = []
        current_id = agent_id
        
        while current_id:
            entry = self.get_by_agent(current_id)
            if entry:
                lineage.append(entry)
                # 获取父代（取第一个）
                if entry.certificate.parent_ids:
                    current_id = entry.certificate.parent_ids[0]
                else:
                    break
            else:
                break
        
        return lineage
    
    def get_descendants(self, agent_id: str) -> List[GraveyardEntry]:
        """
        获取Agent的所有后代
        
        Args:
            agent_id: Agent ID
            
        Returns:
            后代条目列表
        """
        descendants = []
        
        for entry in self._entries:
            if agent_id in entry.certificate.parent_ids:
                descendants.append(entry)
                # 递归获取后代的后代
                descendants.extend(
                    self.get_descendants(entry.certificate.agent_id)
                )
        
        return descendants
    
    def compute_percentiles(self) -> None:
        """
        计算所有条目的百分位数
        
        这会更新每个条目的survival_percentile和pnl_percentile
        """
        # 按代分组
        for gen, indices in self._generation_index.items():
            if len(indices) < 2:
                continue
            
            entries = [self._entries[i] for i in indices]
            
            # 排序并计算百分位
            lifespans = sorted([e.certificate.lifespan for e in entries])
            pnls = sorted([e.certificate.total_pnl for e in entries])
            
            for entry in entries:
                # 存活百分位
                rank = lifespans.index(entry.certificate.lifespan)
                entry.survival_percentile = rank / (len(lifespans) - 1) if len(lifespans) > 1 else 0.5
                
                # 盈亏百分位
                rank = pnls.index(entry.certificate.total_pnl)
                entry.pnl_percentile = rank / (len(pnls) - 1) if len(pnls) > 1 else 0.5
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self._stats_cache_valid and self._stats_cache:
            return self._stats_cache
        
        if not self._entries:
            return {"total_deaths": 0}
        
        # 按死因统计
        cause_counts = {}
        for cause, indices in self._cause_index.items():
            cause_counts[cause.name] = len(indices)
        
        # 按代统计
        generation_counts = {
            gen: len(indices) 
            for gen, indices in self._generation_index.items()
        }
        
        # 寿命统计
        lifespans = [e.certificate.lifespan for e in self._entries]
        avg_lifespan = sum(lifespans) / len(lifespans)
        
        # 盈亏统计
        pnls = [e.certificate.total_pnl for e in self._entries]
        avg_pnl = sum(pnls) / len(pnls)
        
        # 后代统计
        offspring = [e.certificate.offspring_count for e in self._entries]
        avg_offspring = sum(offspring) / len(offspring)
        
        self._stats_cache = {
            "total_deaths": len(self._entries),
            "causes": cause_counts,
            "generations": generation_counts,
            "avg_lifespan": avg_lifespan,
            "max_lifespan": max(lifespans),
            "min_lifespan": min(lifespans),
            "avg_pnl": avg_pnl,
            "max_pnl": max(pnls),
            "min_pnl": min(pnls),
            "avg_offspring": avg_offspring,
            "total_offspring": sum(offspring),
        }
        self._stats_cache_valid = True
        
        return self._stats_cache
    
    def get_recent(self, n: int) -> List[GraveyardEntry]:
        """
        获取最近N个死亡条目
        
        Args:
            n: 数量
            
        Returns:
            条目列表（最近的在最后）
        """
        return self._entries[-n:] if n > 0 else []
    
    def get_extinction_report(self) -> Dict[str, Any]:
        """
        获取灭绝报告
        
        分析哪些DNA谱系已经灭绝
        """
        # 收集所有DNA ID
        all_dna_ids = set()
        parent_dna_ids = set()
        
        for entry in self._entries:
            dna_id = entry.certificate.dna_archive.get("dna_id")
            if dna_id:
                all_dna_ids.add(dna_id)
            
            parent_ids = entry.certificate.parent_ids
            parent_dna_ids.update(parent_ids)
        
        # 没有后代的DNA就是灭绝的谱系
        # （简化实现，实际需要与活着的Agent比较）
        
        return {
            "total_unique_dna": len(all_dna_ids),
            "total_parent_dna": len(parent_dna_ids),
        }
    
    def save_to_disk(self) -> None:
        """保存到磁盘"""
        if not self._persist_path:
            return
        
        os.makedirs(self._persist_path, exist_ok=True)
        
        filename = os.path.join(self._persist_path, "graveyard.jsonl")
        with open(filename, "w", encoding="utf-8") as f:
            for entry in self._entries:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
    
    def __len__(self) -> int:
        return len(self._entries)
    
    def __repr__(self) -> str:
        return f"Graveyard(entries={len(self._entries)})"
