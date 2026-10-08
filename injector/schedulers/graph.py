from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, AbstractSet, Optional

from injector.schedulers.base import FaultScheduler

if TYPE_CHECKING:
    from injector.base import Fault


@dataclass(frozen=True)
class _TriggerRule:
    target_tool: str
    fault: Fault
    trigger_on: str


class GraphFaultScheduler(FaultScheduler):
    """
    Cross-tool trigger graph: a tool's fault is armed by calls to ANOTHER
    tool, letting you express interaction logic across the whole workflow.

        scheduler.register_fault("x1", TimeoutFault(...), trigger_on="x2")

    Semantics (state machine per target):
      - any call to the trigger tool "x2"  -> arm the fault for "x1"
      - next call to "x1" while armed      -> fault fires, then disarm
      - calls to "x1" while unarmed        -> run clean
      - repeatable: each "x2" call re-arms "x1"
    """

    def __init__(self) -> None:
        self._rules: dict[str, list[_TriggerRule]] = {}
        self._armed: set[str] = set()

    def register_fault(self, target_tool: str, fault: Fault, *, trigger_on: str, **_) -> None:
        if trigger_on == target_tool:
            raise ValueError("trigger_on must differ from the target tool.")
        self._rules.setdefault(target_tool, []).append(
            _TriggerRule(target_tool=target_tool, fault=fault, trigger_on=trigger_on)
        )

    def unregister_fault(self, target_tool: str) -> None:
        self._rules.pop(target_tool, None)
        self._armed.discard(target_tool)

    def clear_faults(self) -> None:
        self._rules.clear()
        self._armed.clear()

    def on_tool_call(self, tool_name: str) -> Optional[Fault]:
        if tool_name in self._rules:
            if tool_name in self._armed:
                self._armed.discard(tool_name)
                return self._rules[tool_name][0].fault
            return None
        # Pure trigger observation: arm every rule waiting on this tool.
        for target, rules in self._rules.items():
            if any(rule.trigger_on == tool_name for rule in rules):
                self._armed.add(target)
        return None

    def targets(self) -> AbstractSet[str]:
        return set(self._rules.keys())

    def watches(self) -> AbstractSet[str]:
        triggers = {rule.trigger_on for rules in self._rules.values() for rule in rules}
        return set(self._rules.keys()) | triggers

    def reset(self) -> None:
        self._armed.clear()

    def prediction_state(self) -> frozenset[str]:
        return frozenset()

    def predict(self, state: frozenset[str], tool_name: str) -> tuple[frozenset[str], Optional[Fault]]:
        if tool_name in self._rules:
            if tool_name in state:
                return state - {tool_name}, self._rules[tool_name][0].fault
            return state, None
        triggers = {target for target, rules in self._rules.items()
                    if any(rule.trigger_on == tool_name for rule in rules)}
        return state | triggers, None
