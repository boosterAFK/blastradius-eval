from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol

from injector.observation import ToolObserver

if TYPE_CHECKING:
    from injector.base import Fault


class FaultDispatcher(Protocol):
    """
    Minimal contract the ChaosToolProxy needs from the orchestrator: given
    an observed tool call, return the fault to apply or None.
    """

    def dispatch(self, tool_name: str) -> Optional[Fault]: ...


class ChaosToolProxy:
    def __init__(
        self,
        tool_name: str,
        original_callable: Callable[..., Any],
        dispatcher: FaultDispatcher,
        observer: ToolObserver | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.original_callable = original_callable
        self.dispatcher = dispatcher
        self.observer = observer

    def __call__(self, *args, **kwargs):
        return self.invoke_with_context(None, *args, **kwargs)

    def invoke_with_context(self, context: object | None, *args, **kwargs):
        """Run the faulted callable; context is opaque to the generic injector."""
        if self.observer is None:
            return self._execute(None, *args, **kwargs)
        with self.observer.observe(self.tool_name, args, kwargs, context) as observation:
            return self._execute(observation, *args, **kwargs)

    def _execute(self, observation, *args, **kwargs):
        fault = self.dispatcher.dispatch(self.tool_name)
        if observation is not None:
            observation.record_fault(fault)
        if fault is not None:
            return fault.apply(self.tool_name, self.original_callable, *args, **kwargs)
        return self.original_callable(*args, **kwargs)

    def __repr__(self) -> str:
        return f"ChaosToolProxy(tool_name={self.tool_name!r}, original={self.original_callable!r})"
