"""
世界时钟 - 不可逆的时间系统

核心原则:
- 时间只能向前，永不后退
- 所有Agent共享同一个时钟
- 时钟状态一旦推进，无法回滚
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Callable, Any
from datetime import datetime


@dataclass
class TickEvent:
    """每个时刻的事件记录"""
    tick: int
    timestamp: datetime
    events: List[dict] = field(default_factory=list)


class WorldClock:
    """
    不可逆的世界时钟
    
    这是整个系统的心跳，所有事件都在时钟的驱动下发生。
    时钟一旦前进，就无法回退 —— 这是进化系统的基本约束。
    """
    
    def __init__(self, start_tick: int = 0):
        """
        初始化世界时钟
        
        Args:
            start_tick: 起始时刻（默认为0）
        """
        self._current_tick: int = start_tick
        self._start_time: datetime = datetime.now()
        self._tick_history: List[TickEvent] = []
        self._observers: List[Callable[[int], None]] = []
        self._frozen: bool = False
    
    @property
    def current_tick(self) -> int:
        """获取当前时刻"""
        return self._current_tick
    
    @property
    def elapsed_ticks(self) -> int:
        """已经过的时刻数"""
        return self._current_tick
    
    def tick(self) -> int:
        """
        推进时间一个刻度
        
        这是不可逆的操作！一旦调用，世界就向前推进了。
        
        Returns:
            新的时刻值
        """
        if self._frozen:
            raise RuntimeError("时钟已冻结，无法继续推进（世界末日？）")
        
        # 记录这一刻
        tick_event = TickEvent(
            tick=self._current_tick,
            timestamp=datetime.now()
        )
        self._tick_history.append(tick_event)
        
        # 时间向前，不可逆转
        self._current_tick += 1
        
        # 通知所有观察者
        for observer in self._observers:
            observer(self._current_tick)
        
        return self._current_tick
    
    def tick_multiple(self, n: int) -> int:
        """
        推进时间多个刻度
        
        Args:
            n: 要推进的刻度数
            
        Returns:
            最终的时刻值
        """
        for _ in range(n):
            self.tick()
        return self._current_tick
    
    def register_observer(self, callback: Callable[[int], None]) -> None:
        """
        注册时钟观察者
        
        每次时钟推进时，都会调用注册的回调函数。
        
        Args:
            callback: 接收当前tick的回调函数
        """
        self._observers.append(callback)
    
    def unregister_observer(self, callback: Callable[[int], None]) -> None:
        """移除时钟观察者"""
        if callback in self._observers:
            self._observers.remove(callback)
    
    def freeze(self) -> None:
        """
        冻结时钟（世界末日）
        
        一旦冻结，时钟将无法再推进。
        这用于终止模拟或处理灾难性事件。
        """
        self._frozen = True
    
    def is_frozen(self) -> bool:
        """检查时钟是否已冻结"""
        return self._frozen
    
    def get_history(self, start: int = 0, end: int | None = None) -> List[TickEvent]:
        """
        获取历史记录（只读）
        
        注意：这只是用于观察和分析，不能用于修改历史！
        
        Args:
            start: 起始时刻
            end: 结束时刻（不包含）
            
        Returns:
            历史事件列表
        """
        if end is None:
            end = len(self._tick_history)
        return self._tick_history[start:end]
    
    def __repr__(self) -> str:
        status = "FROZEN" if self._frozen else "RUNNING"
        return f"WorldClock(tick={self._current_tick}, status={status})"


# 全局时钟单例（可选使用）
_global_clock: WorldClock | None = None


def get_global_clock() -> WorldClock:
    """获取全局时钟单例"""
    global _global_clock
    if _global_clock is None:
        _global_clock = WorldClock()
    return _global_clock


def reset_global_clock(start_tick: int = 0) -> WorldClock:
    """
    重置全局时钟（仅用于新模拟实验开始）
    
    警告：这会销毁之前的所有时间记录！
    """
    global _global_clock
    _global_clock = WorldClock(start_tick)
    return _global_clock
