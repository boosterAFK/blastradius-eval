from frameworks.base import FrameworkProvider
from frameworks.langgraph.runner import LangGraphRunner
from injector.adapters.base import ToolAdapter
from frameworks.langgraph.tool_adapter import LangGraphToolAdapter
from telemetry.evaluator import Evaluator

class LangGraphProvider(FrameworkProvider):

    def make_tool_adapter(self) -> "ToolAdapter":
        return LangGraphToolAdapter()

    def make_runner(self, uncompiled_graph, checkpointer=None) -> "LangGraphRunner":
        return LangGraphRunner(uncompiled_graph, checkpointer=checkpointer)

    def make_evaluator(self, instrumentation) -> Evaluator:
        return Evaluator(instrumentation)