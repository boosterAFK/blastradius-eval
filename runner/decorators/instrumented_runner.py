from typing import Any, Dict, Optional
from uuid import uuid4

from runner.base import AgentRunner
from telemetry.instrumentation import Instrumentation
from telemetry.span import SpanKind


class InstrumentedRunner:
    def __init__(
        self,
        runner: AgentRunner,
        instrumentation: Instrumentation,
        agent_name: str = "default-agent",
        agent_id: Optional[str] = None,
    ) -> None:
        self.runner = runner
        self.instrumentation = instrumentation
        self.agent_name = agent_name
        self.agent_id = agent_id if agent_id is not None else str(uuid4())

    def invoke(
        self,
        input_data: Dict[str, Any],
        thread_id: str,
        recursion_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        attrs = {
            "gen_ai.agent.name": self.agent_name,
            "gen_ai.agent.id": self.agent_id,
            "agent.run.id": str(uuid4()),
            "agent.thread_id": thread_id,
        }
        if recursion_limit is not None:
            attrs["agent.recursion_limit"] = recursion_limit

        with self.instrumentation.span_context(
            name=f"{self.agent_name}.invoke",
            span_kind=SpanKind.AGENT,
            **attrs,
        ):
            return self.runner.invoke(
                input_data,
                thread_id=thread_id,
                recursion_limit=recursion_limit,
            )

    def compile(self) -> Any:
        return self.runner.compile()

    def get_state(self, thread_id: str) -> Dict[str, Any]:
        return self.runner.get_state(thread_id)

    def update_state(
        self,
        thread_id: str,
        new_state: Dict[str, Any],
        as_node: Optional[str] = None,
    ) -> None:
        self.runner.update_state(thread_id, new_state, as_node=as_node)
