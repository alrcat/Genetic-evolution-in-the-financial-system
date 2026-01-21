"""
快照存储 - 世界状态的定期备份

核心原则:
- 快照是只读的
- 用于快速恢复到某个时间点进行分析
- 不能用于修改历史
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import os
import gzip


@dataclass
class WorldSnapshot:
    """
    世界快照
    
    记录某一时刻的完整世界状态
    """
    
    snapshot_id: str
    tick: int
    created_at: datetime
    
    # 环境状态
    market_state: Dict[str, Any] = field(default_factory=dict)
    
    # 种群状态
    population_size: int = 0
    alive_count: int = 0
    dead_count: int = 0
    
    # 代际分布
    generation_distribution: Dict[int, int] = field(default_factory=dict)
    
    # DNA多样性
    unique_dna_count: int = 0
    
    # 资源统计
    total_capital: float = 0.0
    average_capital: float = 0.0
    
    # Agent摘要列表
    agent_summaries: List[Dict[str, Any]] = field(default_factory=list)
    
    # 基因频率
    gene_frequencies: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "snapshot_id": self.snapshot_id,
            "tick": self.tick,
            "created_at": self.created_at.isoformat(),
            "market_state": self.market_state,
            "population_size": self.population_size,
            "alive_count": self.alive_count,
            "dead_count": self.dead_count,
            "generation_distribution": self.generation_distribution,
            "unique_dna_count": self.unique_dna_count,
            "total_capital": self.total_capital,
            "average_capital": self.average_capital,
            "agent_summaries": self.agent_summaries,
            "gene_frequencies": self.gene_frequencies,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorldSnapshot:
        """从字典反序列化"""
        return cls(
            snapshot_id=data["snapshot_id"],
            tick=data["tick"],
            created_at=datetime.fromisoformat(data["created_at"]),
            market_state=data.get("market_state", {}),
            population_size=data.get("population_size", 0),
            alive_count=data.get("alive_count", 0),
            dead_count=data.get("dead_count", 0),
            generation_distribution=data.get("generation_distribution", {}),
            unique_dna_count=data.get("unique_dna_count", 0),
            total_capital=data.get("total_capital", 0.0),
            average_capital=data.get("average_capital", 0.0),
            agent_summaries=data.get("agent_summaries", []),
            gene_frequencies=data.get("gene_frequencies", {}),
        )
    
    def __repr__(self) -> str:
        return (
            f"WorldSnapshot(t={self.tick}, "
            f"pop={self.alive_count}/{self.population_size}, "
            f"dna={self.unique_dna_count})"
        )


class SnapshotStore:
    """
    快照存储
    
    管理世界状态快照，支持：
    1. 定期自动快照
    2. 手动快照
    3. 快照查询
    4. 压缩存储
    """
    
    def __init__(
        self,
        persist_path: Optional[str] = None,
        auto_snapshot_interval: int = 1000,  # 每N个tick自动快照
        max_snapshots_in_memory: int = 100,
        compress: bool = True,
    ):
        """
        初始化快照存储
        
        Args:
            persist_path: 持久化路径
            auto_snapshot_interval: 自动快照间隔
            max_snapshots_in_memory: 内存中保留的最大快照数
            compress: 是否压缩存储
        """
        self._snapshots: List[WorldSnapshot] = []
        self._tick_index: Dict[int, int] = {}  # tick -> index
        
        self._persist_path = persist_path
        self._auto_interval = auto_snapshot_interval
        self._max_in_memory = max_snapshots_in_memory
        self._compress = compress
        
        self._snapshot_counter: int = 0
    
    def create_snapshot(
        self,
        tick: int,
        environment: Any,  # Environment
        agents: List[Any],  # List[Agent]
    ) -> WorldSnapshot:
        """
        创建快照
        
        Args:
            tick: 当前时刻
            environment: 环境对象
            agents: Agent列表
            
        Returns:
            创建的快照
        """
        import uuid
        
        # 收集Agent统计
        alive_agents = [a for a in agents if a.is_alive]
        dead_agents = [a for a in agents if not a.is_alive]
        
        # 代际分布
        gen_dist: Dict[int, int] = {}
        for agent in alive_agents:
            gen = agent.lineage.generation
            gen_dist[gen] = gen_dist.get(gen, 0) + 1
        
        # DNA多样性
        unique_dna = set(a.dna.dna_id for a in alive_agents)
        
        # 资本统计
        total_capital = sum(a.resources.capital for a in alive_agents)
        avg_capital = total_capital / len(alive_agents) if alive_agents else 0
        
        # Agent摘要
        summaries = [a.get_summary() for a in alive_agents[:100]]  # 限制数量
        
        # 基因频率（简化）
        gene_freq: Dict[str, float] = {}
        total_genes = 0
        for agent in alive_agents:
            for gene in agent.dna.get_all_genes():
                gene_freq[gene.gene_type.name] = gene_freq.get(gene.gene_type.name, 0) + 1
                total_genes += 1
        
        if total_genes > 0:
            gene_freq = {k: v/total_genes for k, v in gene_freq.items()}
        
        # 创建快照
        snapshot = WorldSnapshot(
            snapshot_id=str(uuid.uuid4()),
            tick=tick,
            created_at=datetime.now(),
            market_state=environment.get_market_summary() if hasattr(environment, 'get_market_summary') else {},
            population_size=len(agents),
            alive_count=len(alive_agents),
            dead_count=len(dead_agents),
            generation_distribution=gen_dist,
            unique_dna_count=len(unique_dna),
            total_capital=total_capital,
            average_capital=avg_capital,
            agent_summaries=summaries,
            gene_frequencies=gene_freq,
        )
        
        self._store_snapshot(snapshot)
        return snapshot
    
    def _store_snapshot(self, snapshot: WorldSnapshot) -> None:
        """存储快照"""
        self._snapshots.append(snapshot)
        self._tick_index[snapshot.tick] = len(self._snapshots) - 1
        self._snapshot_counter += 1
        
        # 内存管理
        if len(self._snapshots) > self._max_in_memory:
            self._flush_old_snapshots()
    
    def _flush_old_snapshots(self) -> None:
        """将旧快照写入磁盘"""
        if not self._persist_path:
            # 没有持久化路径，直接丢弃
            keep_count = self._max_in_memory // 2
            self._snapshots = self._snapshots[-keep_count:]
            self._rebuild_index()
            return
        
        os.makedirs(self._persist_path, exist_ok=True)
        
        # 保留最新的一半
        keep_count = self._max_in_memory // 2
        to_flush = self._snapshots[:-keep_count]
        self._snapshots = self._snapshots[-keep_count:]
        
        # 写入磁盘
        for snapshot in to_flush:
            self._write_snapshot_to_disk(snapshot)
        
        self._rebuild_index()
    
    def _write_snapshot_to_disk(self, snapshot: WorldSnapshot) -> None:
        """将快照写入磁盘"""
        if not self._persist_path:
            return
        
        filename = f"snapshot_{snapshot.tick}_{snapshot.snapshot_id[:8]}.json"
        if self._compress:
            filename += ".gz"
        
        filepath = os.path.join(self._persist_path, filename)
        
        data = json.dumps(snapshot.to_dict(), ensure_ascii=False)
        
        if self._compress:
            with gzip.open(filepath, "wt", encoding="utf-8") as f:
                f.write(data)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(data)
    
    def _rebuild_index(self) -> None:
        """重建索引"""
        self._tick_index = {
            s.tick: i for i, s in enumerate(self._snapshots)
        }
    
    def get_by_tick(self, tick: int) -> Optional[WorldSnapshot]:
        """按tick获取快照"""
        index = self._tick_index.get(tick)
        if index is not None:
            return self._snapshots[index]
        
        # 尝试从磁盘加载
        if self._persist_path:
            return self._load_snapshot_from_disk(tick)
        
        return None
    
    def _load_snapshot_from_disk(self, tick: int) -> Optional[WorldSnapshot]:
        """从磁盘加载快照"""
        if not self._persist_path or not os.path.exists(self._persist_path):
            return None
        
        # 查找匹配的文件
        for filename in os.listdir(self._persist_path):
            if filename.startswith(f"snapshot_{tick}_"):
                filepath = os.path.join(self._persist_path, filename)
                
                if filename.endswith(".gz"):
                    with gzip.open(filepath, "rt", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                
                return WorldSnapshot.from_dict(data)
        
        return None
    
    def get_nearest(self, tick: int) -> Optional[WorldSnapshot]:
        """获取最接近的快照"""
        if not self._snapshots:
            return None
        
        # 在内存快照中查找
        nearest = min(self._snapshots, key=lambda s: abs(s.tick - tick))
        return nearest
    
    def get_range(
        self, 
        start_tick: int, 
        end_tick: int
    ) -> List[WorldSnapshot]:
        """获取时间范围内的所有快照"""
        return [
            s for s in self._snapshots
            if start_tick <= s.tick < end_tick
        ]
    
    def should_snapshot(self, tick: int) -> bool:
        """检查是否应该创建快照"""
        return tick % self._auto_interval == 0
    
    def get_timeline(self) -> List[Dict[str, Any]]:
        """获取快照时间线摘要"""
        return [
            {
                "tick": s.tick,
                "alive": s.alive_count,
                "generations": len(s.generation_distribution),
                "dna_diversity": s.unique_dna_count,
                "avg_capital": s.average_capital,
            }
            for s in self._snapshots
        ]
    
    def __len__(self) -> int:
        return len(self._snapshots)
    
    def __repr__(self) -> str:
        return f"SnapshotStore(snapshots={len(self._snapshots)}, total={self._snapshot_counter})"
