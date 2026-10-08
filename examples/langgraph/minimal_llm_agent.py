from typing import Annotated, List, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def fetch_data(query: str) -> str:
    """Fetches system data."""
    result = f"Data for {query}"
    _fetches.append((query, result))
    return result


_fetches: list[tuple[str, str]] = []


def reset_world() -> None:
    _fetches.clear()


def world_snapshot() -> tuple[tuple[str, str], ...]:
    """Private task evidence, independent of agent-authored messages."""
    return tuple(_fetches)


@tool
def report_unavailable(reason: str) -> str:
    """Use this when fetch_data fails and cannot be retried."""
    return f"Unavailable: {reason}"


# Marks the tool as a graceful terminal path (read by the evaluator via the
# ToolMessage\'s additional_kwargs). The @tool decorator has no metadata
# kwarg, so it is set on the StructuredTool directly; the poison adapter\'s
# model_copy carries it onto poisoned clones.
report_unavailable.metadata = {"terminal": True}

# Every tool this workflow can execute. Poisoned clones are injected at
# build_workflow() time, not here.
TOOLS: List[BaseTool] = [fetch_data, report_unavailable]

# The task this workflow is benchmarked against. Owned by the workflow
# catalog; run_benchmark.py feeds it to the runner.
PROMPT = (
    "Fetch the system data."
)

load_dotenv()

llm = ChatOpenAI(model="gpt-5", temperature=2)


def build_workflow(tools: List[BaseTool]) -> StateGraph:
    """
    Builds the (uncompiled) workflow around the GIVEN tool objects. The LLM
    binds to these tools and the tool node executes exactly these instances,
    so passing poisoned clones here is all the runner ever needs.
    """
    tools_registry = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    def llm_node(state: AgentState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def tool_node(state: AgentState):
        responses = []
        for tool_call in state["messages"][-1].tool_calls:
            tool_instance = tools_registry.get(tool_call["name"])

            if not tool_instance:
                responses.append(ToolMessage(
                    content=f"Error: Tool {tool_call['name']} not found.",
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                    status="error"
                ))
                continue

            try:
                result = tool_instance.invoke(tool_call["args"])
                responses.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                    name=tool_instance.name,
                    additional_kwargs={"terminal": bool((tool_instance.metadata or {}).get("terminal"))}))
            except Exception as e:
                responses.append(ToolMessage(
                    content=f"Error: {e}",
                    tool_call_id=tool_call["id"],
                    name=tool_instance.name,
                    status="error"
                ))

        return {"messages": responses}

    def router(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("llm", llm_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("llm")
    workflow.add_conditional_edges("llm", router, {"tools": "tools", END: END})
    workflow.add_edge("tools", "llm")
    return workflow
