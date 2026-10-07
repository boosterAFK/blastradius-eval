from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, List, Optional, Protocol

from injector.adapters.base import ToolAdapter
from injector.chaos_tool_proxy import ChaosToolProxy
from injector.schedulers.base import FaultScheduler

if TYPE_CHECKING:
    from telemetry.instrumentation import Instrumentation


class Fault(ABC):
    @abstractmethod
    def apply(self, tool_name: str, original_callable: Callable, *args, **kwargs) -> Any:
        pass


class FaultInjector:

    def __init__(
        self,
        adapter: ToolAdapter,
        schedulers: Optional[List[FaultScheduler]] = None,
        instrumentation: Optional[Instrumentation] = None,
       
    ):
        self._schedulers: List[FaultScheduler] = []
        self._instrumentation = instrumentation
        for scheduler in schedulers or []:
            self.add_scheduler(scheduler)
        self._adapter = adapter 

    @property
    def schedulers(self) -> tuple[FaultScheduler, ...]:
        return tuple(self._schedulers)

    @property
    def instrumentation(self) -> Optional[Instrumentation]:
        return self._instrumentation

    def add_scheduler(self, scheduler: FaultScheduler) -> None:
        overlap = scheduler.targets() & self._owned_targets()
        if overlap:
            raise ValueError(
                f"Scheduler target conflict: tools {sorted(overlap)} are already "
                "owned by another registered scheduler. Each tool must have "
                "exactly one fault owner - compose behaviour explicitly instead."
            )
        self._schedulers.append(scheduler)

    def reset_run(self) -> None:
        for scheduler in self._schedulers:
            scheduler.reset()

    def dispatch(self, tool_name: str) -> Optional[Fault]:
        for scheduler in self._schedulers:
            if tool_name not in scheduler.watches():
                continue
            fault = scheduler.on_tool_call(tool_name)
            if fault is not None:
                return fault
        return None

    def poison_tools(self, tools: List[Any]) -> List[Any]:
        watched = set()
        for scheduler in self._schedulers:
            watched |= scheduler.watches()

        poisoned: List[Any] = []
        for tool in tools:
            name = self._adapter.get_name(tool)
            if name not in watched:
                poisoned.append(tool)
                continue
            observer = self._instrumentation.make_tool_observer() if self._instrumentation is not None else None
            proxy = ChaosToolProxy(name, self._adapter.get_callable(tool), self, observer=observer)
            poisoned.append(self._adapter.clone_with_callable(tool, proxy))
        return poisoned

    def _owned_targets(self) -> set:
        owned: set = set()
        for scheduler in self._schedulers:
            owned |= scheduler.targets()
        return owned
