from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AbstractSet, Optional, Any

if TYPE_CHECKING:
    from injector.base import Fault


class FaultScheduler(ABC):
    """
    Base interface for pluggable fault-scheduling strategies.

    A scheduler encapsulates HOW and WHEN faults fire for the tools it owns.
    The orchestrator (FaultInjector) communicates with schedulers exclusively
    through this interface:

      - ``poison_tools`` consults ``watches()`` to decide which tools must be
        wrapped with a ChaosToolProxy.
      - every observed tool call is routed to ``on_tool_call`` before the
        underlying tool executes; the scheduler inspects its internal state
        (FSM, timeline, rule graph, ...) and returns the Fault to apply, or
        None to let the call run clean.

    Implementations own their arming logic: strategy-specific parameters
    (e.g. ``at_calls`` for FixedCallScheduler, ``trigger_on`` for
    GraphFaultScheduler) are passed to that scheduler's ``register_fault``,
    never to the orchestrator.
    """

    @abstractmethod
    def register_fault(self, target_tool: str, fault: Fault, **kwargs) -> None:
        """Arm `fault` for `target_tool` using this scheduler's own semantics."""

    @abstractmethod
    def unregister_fault(self, target_tool: str) -> None:
        """Remove all armed logic for `target_tool`."""

    @abstractmethod
    def clear_faults(self) -> None:
        """Remove all armed logic for every tool."""

    @abstractmethod
    def on_tool_call(self, tool_name: str) -> Optional[Fault]:
        """
        Observe an incoming call to `tool_name` and decide its fate.

        Implementations decide using ONLY past state, return the Fault to
        apply (or None), and then record this call into their internal state.
        """

    @abstractmethod
    def targets(self) -> AbstractSet[str]:
        """Tools this scheduler may inject faults into."""

    def watches(self) -> AbstractSet[str]:
        """
        Tools whose calls this scheduler needs to observe. Defaults to
        ``targets()``; schedulers with cross-tool logic (e.g.
        GraphFaultScheduler) override this to include pure trigger tools,
        which get reporting proxies even though they never fault themselves.
        """
        return self.targets()

    @abstractmethod
    def reset(self) -> None:
        """
        Clear runtime state between benchmark runs (counters, armed flags,
        FSM positions). The armed schedule/rules themselves are preserved.
        """

    @abstractmethod
    def prediction_state(self) -> Any:
        """Return fresh, isolated state for a hypothetical run."""

    @abstractmethod
    def predict(self, state: Any, tool_name: str) -> tuple[Any, Optional[Fault]]:
        """Return next hypothetical state and fault without changing runtime state."""
