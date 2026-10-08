from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, AbstractSet, Optional

from injector.schedulers.base import FaultScheduler

if TYPE_CHECKING:
    from injector.base import Fault


class FixedCallScheduler(FaultScheduler):
    """
    Deterministic call-count timeline: faults are pre-computed against the
    Nth call of each tool.

        scheduler.register_fault("fetch_data", TimeoutFault(...), at_calls=(1, 2))
    """

    def __init__(self) -> None:
        self._timeline: dict[str, list[tuple[int, Fault]]] = {}
        self._counters: defaultdict[str, int] = defaultdict(int)

    def register_fault(self, target_tool: str, fault: Fault, *, at_calls: tuple[int, ...] = (1,), **_) -> None:
        entries = self._timeline.setdefault(target_tool, [])
        for call_number in at_calls:
            if call_number < 1:
                raise ValueError(f"at_calls entries must be >= 1 (got {call_number}).")
            entries.append((call_number, fault))
        entries.sort(key=lambda entry: entry[0])

    def unregister_fault(self, target_tool: str) -> None:
        self._timeline.pop(target_tool, None)

    def clear_faults(self) -> None:
        self._timeline.clear()

    def on_tool_call(self, tool_name: str) -> Optional[Fault]:
        self._counters[tool_name] += 1
        step = self._counters[tool_name]
        for call_number, fault in self._timeline.get(tool_name, ()):
            if call_number == step:
                return fault
            if call_number > step:
                break
        return None

    def targets(self) -> AbstractSet[str]:
        return set(self._timeline.keys())

    def step_of(self, tool_name: str) -> int:
        return self._counters.get(tool_name, 0)

    def reset(self) -> None:
        self._counters.clear()

    def prediction_state(self) -> dict[str, int]:
        return {}

    def predict(self, state: dict[str, int], tool_name: str) -> tuple[dict[str, int], Optional[Fault]]:
        next_state = {**state, tool_name: state.get(tool_name, 0) + 1}
        fault = next((fault for call, fault in self._timeline.get(tool_name, ())
                      if call == next_state[tool_name]), None)
        return next_state, fault
