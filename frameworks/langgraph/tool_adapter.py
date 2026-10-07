from inspect import signature

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools.base import _get_runnable_config_param

from injector.adapters.base import ToolAdapter
from injector.chaos_tool_proxy import ChaosToolProxy
from frameworks.langgraph.tool_span import get_current_tool_span

class LangGraphToolAdapter(ToolAdapter):
    def get_name(self, tool):            return tool.name
    def get_callable(self, tool):        return tool.func
    def clone_with_callable(self, tool, fn): 
        if getattr(tool, "func", None) is None:
            raise TypeError(
                f"Cannot poison tool '{tool.name}': only langchain_core.tools.Tool "
                "instances created from a function (e.g. via @tool) are supported."
            )
        if isinstance(fn, ChaosToolProxy):
            # StructuredTool passes the current run's child CallbackManager
            # only to callables declaring `callbacks`. This is a framework
            # detail: pass it through only if the original tool requested it.
            original_accepts_callbacks = "callbacks" in signature(fn.original_callable).parameters
            original_config_param = _get_runnable_config_param(fn.original_callable)

            def invoke_proxy(*args, callbacks=None, config: RunnableConfig = None, **kwargs):
                if original_accepts_callbacks and callbacks is not None:
                    kwargs["callbacks"] = callbacks
                if original_config_param is not None and config is not None:
                    kwargs[original_config_param] = config
                span = get_current_tool_span(callbacks)
                return fn.invoke_with_context(span, *args, **kwargs)

            return tool.model_copy(update={"func": invoke_proxy})
        return tool.model_copy(update={"func": fn})
