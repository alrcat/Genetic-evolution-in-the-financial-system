"""
存储层 - 事件溯源与历史记录

核心原则:
- 所有状态变化都是事件
- 事件一旦写入不可修改
- 时间不可逆
"""

from storage.event_store import EventStore, Event, EventType
from storage.graveyard import Graveyard
from storage.snapshot import SnapshotStore

__all__ = [
    "EventStore",
    "Event",
    "EventType",
    "Graveyard",
    "SnapshotStore",
]
