"""
事件存储 - 不可变的历史记录

核心原则:
- 所有发生的事都记录为事件
- 事件一旦写入不可修改
- 这是进化历史的真实记录
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional
import json
import os


class EventType(Enum):
    """事件类型枚举"""
    # 生命周期事件
    AGENT_BORN = auto()
    AGENT_DIED = auto()
    
    # 繁殖事件
    REPRODUCTION_ATTEMPTED = auto()
    REPRODUCTION_SUCCEEDED = auto()
    
    # 交易事件
    ORDER_PLACED = auto()
    TRADE_EXECUTED = auto()
    POSITION_CLOSED = auto()
    
    # 基因事件
    GENE_MUTATED = auto()
    GENE_EXPRESSED = auto()
    DORMANT_GENE_ACTIVATED = auto()
    
    # 环境事件
    REGIME_CHANGED = auto()
    MARKET_TICK = auto()
    EXTINCTION_EVENT = auto()
    
    # 系统事件
    SIMULATION_STARTED = auto()
    SIMULATION_ENDED = auto()
    SNAPSHOT_CREATED = auto()


@dataclass
class Event:
    """
    事件 - 不可变的历史记录
    
    每个事件代表系统中发生的一件事。
    事件一旦创建就不能修改。
    """
    
    event_id: str
    event_type: EventType
    timestamp: int           # 世界时钟时刻
    created_at: datetime     # 真实世界时间
    
    # 事件主体
    agent_id: Optional[str] = None
    
    # 事件数据
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # 关联事件
    related_events: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.name,
            "timestamp": self.timestamp,
            "created_at": self.created_at.isoformat(),
            "agent_id": self.agent_id,
            "payload": self.payload,
            "related_events": self.related_events,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Event:
        """从字典反序列化"""
        return cls(
            event_id=data["event_id"],
            event_type=EventType[data["event_type"]],
            timestamp=data["timestamp"],
            created_at=datetime.fromisoformat(data["created_at"]),
            agent_id=data.get("agent_id"),
            payload=data.get("payload", {}),
            related_events=data.get("related_events", []),
        )
    
    def __repr__(self) -> str:
        agent_str = f", agent={self.agent_id[:8]}" if self.agent_id else ""
        return f"Event({self.event_type.name}, t={self.timestamp}{agent_str})"


class EventStore:
    """
    事件存储
    
    所有系统事件的不可变记录。
    支持：
    - 只追加写入
    - 按时间范围查询
    - 按类型查询
    - 按Agent查询
    """
    
    def __init__(
        self, 
        persist_path: Optional[str] = None,
        max_memory_events: int = 100000,
    ):
        """
        初始化事件存储
        
        Args:
            persist_path: 持久化路径（可选）
            max_memory_events: 内存中保留的最大事件数
        """
        self._events: List[Event] = []
        self._event_index: Dict[str, int] = {}  # event_id -> index
        self._type_index: Dict[EventType, List[int]] = {}  # type -> indices
        self._agent_index: Dict[str, List[int]] = {}  # agent_id -> indices
        
        self._persist_path = persist_path
        self._max_memory_events = max_memory_events
        self._event_counter: int = 0
        
        # 事件监听器
        self._listeners: Dict[EventType, List[Callable[[Event], None]]] = {}
    
    def append(self, event: Event) -> None:
        """
        追加事件
        
        事件一旦追加就不可修改或删除。
        
        Args:
            event: 要追加的事件
        """
        index = len(self._events)
        
        # 存储事件
        self._events.append(event)
        
        # 更新索引
        self._event_index[event.event_id] = index
        
        if event.event_type not in self._type_index:
            self._type_index[event.event_type] = []
        self._type_index[event.event_type].append(index)
        
        if event.agent_id:
            if event.agent_id not in self._agent_index:
                self._agent_index[event.agent_id] = []
            self._agent_index[event.agent_id].append(index)
        
        self._event_counter += 1
        
        # 通知监听器
        self._notify_listeners(event)
        
        # 内存管理
        if len(self._events) > self._max_memory_events:
            self._flush_to_disk()
    
    def create_event(
        self,
        event_type: EventType,
        timestamp: int,
        agent_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        related_events: Optional[List[str]] = None,
    ) -> Event:
        """
        创建并追加事件
        
        Args:
            event_type: 事件类型
            timestamp: 世界时钟时刻
            agent_id: 相关Agent ID
            payload: 事件数据
            related_events: 关联事件ID
            
        Returns:
            创建的事件
        """
        import uuid
        
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=timestamp,
            created_at=datetime.now(),
            agent_id=agent_id,
            payload=payload or {},
            related_events=related_events or [],
        )
        
        self.append(event)
        return event
    
    def get_by_id(self, event_id: str) -> Optional[Event]:
        """按ID获取事件"""
        index = self._event_index.get(event_id)
        if index is not None:
            return self._events[index]
        return None
    
    def get_by_type(
        self, 
        event_type: EventType,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Event]:
        """
        按类型获取事件
        
        Args:
            event_type: 事件类型
            start_time: 开始时刻（包含）
            end_time: 结束时刻（不包含）
            
        Returns:
            事件列表
        """
        indices = self._type_index.get(event_type, [])
        events = [self._events[i] for i in indices]
        
        if start_time is not None:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time is not None:
            events = [e for e in events if e.timestamp < end_time]
        
        return events
    
    def get_by_agent(
        self, 
        agent_id: str,
        event_types: Optional[List[EventType]] = None,
    ) -> List[Event]:
        """
        按Agent获取事件
        
        Args:
            agent_id: Agent ID
            event_types: 筛选的事件类型
            
        Returns:
            事件列表
        """
        indices = self._agent_index.get(agent_id, [])
        events = [self._events[i] for i in indices]
        
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        
        return events
    
    def get_by_time_range(
        self,
        start_time: int,
        end_time: int,
        event_types: Optional[List[EventType]] = None,
    ) -> List[Event]:
        """
        按时间范围获取事件
        
        Args:
            start_time: 开始时刻（包含）
            end_time: 结束时刻（不包含）
            event_types: 筛选的事件类型
            
        Returns:
            事件列表
        """
        events = [
            e for e in self._events 
            if start_time <= e.timestamp < end_time
        ]
        
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        
        return events
    
    def get_recent(
        self, 
        n: int, 
        event_types: Optional[List[EventType]] = None
    ) -> List[Event]:
        """
        获取最近N个事件
        
        Args:
            n: 数量
            event_types: 筛选的事件类型
            
        Returns:
            事件列表
        """
        if event_types:
            events = [e for e in self._events if e.event_type in event_types]
        else:
            events = self._events
        
        return events[-n:]
    
    def register_listener(
        self, 
        event_type: EventType, 
        callback: Callable[[Event], None]
    ) -> None:
        """
        注册事件监听器
        
        Args:
            event_type: 要监听的事件类型
            callback: 回调函数
        """
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)
    
    def _notify_listeners(self, event: Event) -> None:
        """通知监听器"""
        listeners = self._listeners.get(event.event_type, [])
        for callback in listeners:
            try:
                callback(event)
            except Exception as e:
                # 监听器错误不应该影响事件存储
                pass
    
    def _flush_to_disk(self) -> None:
        """将旧事件刷新到磁盘"""
        if not self._persist_path:
            # 没有持久化路径，直接丢弃旧事件
            trim_count = len(self._events) - self._max_memory_events // 2
            if trim_count > 0:
                self._events = self._events[trim_count:]
                # 重建索引
                self._rebuild_indices()
            return
        
        # 持久化到磁盘
        os.makedirs(self._persist_path, exist_ok=True)
        
        # 写入文件
        filename = os.path.join(
            self._persist_path, 
            f"events_{self._event_counter}.jsonl"
        )
        
        trim_count = len(self._events) - self._max_memory_events // 2
        events_to_write = self._events[:trim_count]
        
        with open(filename, "w", encoding="utf-8") as f:
            for event in events_to_write:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        
        # 从内存中移除
        self._events = self._events[trim_count:]
        self._rebuild_indices()
    
    def _rebuild_indices(self) -> None:
        """重建索引"""
        self._event_index.clear()
        self._type_index.clear()
        self._agent_index.clear()
        
        for i, event in enumerate(self._events):
            self._event_index[event.event_id] = i
            
            if event.event_type not in self._type_index:
                self._type_index[event.event_type] = []
            self._type_index[event.event_type].append(i)
            
            if event.agent_id:
                if event.agent_id not in self._agent_index:
                    self._agent_index[event.agent_id] = []
                self._agent_index[event.agent_id].append(i)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        type_counts = {
            t.name: len(indices) 
            for t, indices in self._type_index.items()
        }
        
        return {
            "total_events": len(self._events),
            "total_recorded": self._event_counter,
            "unique_agents": len(self._agent_index),
            "event_types": type_counts,
        }
    
    def get_event_counts(self) -> Dict[str, int]:
        """
        获取各类事件的计数
        
        Returns:
            事件类型 -> 数量的字典
        """
        return {
            t.name: len(indices) 
            for t, indices in self._type_index.items()
        }
    
    def __len__(self) -> int:
        return len(self._events)
    
    def __repr__(self) -> str:
        return f"EventStore(events={len(self._events)}, total={self._event_counter})"
