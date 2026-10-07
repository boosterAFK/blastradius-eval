# BlastRadius Eval Suite

> **A headless testing harness that measures how multi-step agents behave when their tools fail.**

Most benchmarks only score the final outcome - did it compile, did the string match. In production,
what kills an agent is its **trajectory**: circular retry loops burning API credits, silent recovery
failures, latency degradation under tool timeouts. BlastRadius Eval Suite subjects agent graphs to
deterministic, pre-computed fault schedules and scores the resulting trajectory.

---

## Quickstart

```bash
# 1. Install dependencies (uv manages the .venv at the repo root)
uv sync

# 2. Add your OpenAI key
cp .env.example .env   # then set OPENAI_API_KEY=...

# 3. Run a scenario
.venv\Scripts\python.exe examples/run_benchmark.py                      # linear fetch + retry
.venv\Scripts\python.exe examples/run_secure_migration_benchmark.py     # branching PCI-migration DAG
```

The linear scenario prints something like:

```
Efficiency: 0.5714285714285714
Recovery Rate: 1.0
Graceful Termination: 1.0
```

The migration scenario classifies the trajectory and reports **two** efficiency numbers:

```
Trajectory: RESILIENT (retried init_transfer in place)
Efficiency (clean baseline): 0.833     # vs. a no-fault world (blast radius)
Efficiency (fault-aware):    1.0       # vs. the best path GIVEN the scheduled faults
Recovery Rate: 1.0
Graceful Termination: 1.0
```

---

## How it works

```
examples/langgraph/                 PURE WORKFLOW CATALOG (test subjects)
  minimal_llm_agent.py                linear: fetch_data + report_unavailable
  secure_migration_agent.py           branching PCI-migration DAG (8 tools)
        |                             each exposes TOOLS + build_workflow + PROMPT
        |                             no imports from injector / runner / frameworks
        v
examples/run_*_benchmark.py         WIRING POINTS (one per scenario)
        |
        |-- FrameworkProvider (Abstract Factory)
        |      -> make_runner() / make_evaluator() / make_tool_adapter()
        |      frameworks/langgraph/: LangGraphProvider
        |
        +-- FaultInjector (dispatcher, framework-agnostic)
               -> poison_tools()  clones tools via the adapter, swaps .func
               -> dispatch()      routes each observed call to the schedulers
                    |
                    +-- FaultScheduler (pluggable strategies, own their logic)
                    |      FixedCallScheduler  -> fire on call N  (at_calls=(1,2))
                    |      GraphFaultScheduler -> x2 call arms x1 (trigger_on="...")
                    |
                    +-- ChaosToolProxy -> becomes the tool\'s func; per-call fault
                                          decision means NO runtime graph mutation
                                          emits OpenInference-tagged OTel spans
```

### The pieces

| Component | Role |
| --- | --- |
| **Workflow catalog** (`examples/langgraph/`) | Pure test subjects. Each module exposes `TOOLS`, `build_workflow(tools)`, and `PROMPT`. They never import chaos code. |
| **FaultScheduler** (`injector/schedulers/`) | Pluggable strategies that decide WHEN a fault fires. Each owns its arming logic and `forced_extra_steps()`. |
| **FaultInjector** (`injector/base.py`) | Framework-agnostic dispatcher. Owns the scheduler set, enforces strict target ownership, produces poisoned tool clones. |
| **ChaosToolProxy** (`injector/chaos_tool_proxy.py`) | Generic callable wrapper installed as the tool\'s `func`; asks the injector on every call whether to fault; emits a tool span. |
| **FrameworkProvider** (`frameworks/`) | Abstract Factory returning a matched family (runner + evaluator + tool adapter) for one framework. |
| **AgentRunner** (`runner/base.py`, `frameworks/langgraph/runner.py`) | Dumb executor. Receives a fully-built (already-poisoned) workflow; never touches tools or faults. |
| **Instrumentation** (`telemetry/instrumentation.py`) | OTel `TracerProvider` with an in-memory exporter (for metrics) and an optional OTLP exporter (Phoenix / collector). |

---

## Writing your own scenario

```python
from frameworks.langgraph import LangGraphProvider
from examples.langgraph.minimal_llm_agent import PROMPT, TOOLS, build_workflow
from injector.base import FaultInjector
from injector.faults import TimeoutFault
from injector.schedulers import FixedCallScheduler
from telemetry.instrumentation import Instrumentation
from telemetry.metrics import StepEfficiencyMetric
from langchain_core.messages import HumanMessage

framework = LangGraphProvider()
instrumentation = Instrumentation(service_name="blastradius-eval", enable_otlp=True)

# 1. Arm a scheduling strategy (it owns its arming logic).
scheduler = FixedCallScheduler()
scheduler.register_fault("fetch_data", TimeoutFault(delay_seconds=0.1), at_calls=(1, 2))

# 2. Poison the tools (originals stay pristine) and build the workflow.
injector = FaultInjector(
    schedulers=[scheduler],
    adapter=framework.make_tool_adapter(),
    instrumentation=instrumentation,
)
workflow = build_workflow(injector.poison_tools(TOOLS))

# 3. Run with a dumb executor and score the trajectory.
runner = framework.make_runner(uncompiled_graph=workflow)
runner.invoke({"messages": [HumanMessage(content=PROMPT)]}, thread_id="chaos-001", recursion_limit=100)
final_state = runner.get_state("chaos-001")
instrumentation.provider.force_flush()

clean = StepEfficiencyMetric(optimal_steps=4, baseline="clean")
aware = StepEfficiencyMetric(optimal_steps=4, scheduler=scheduler, baseline="fault_aware")
print(clean.compute(final_state), aware.compute(final_state))
```

### Cross-tool triggers

`GraphFaultScheduler` expresses interaction logic across tools - e.g. "after
`get_db_creds` is called, the NEXT `init_transfer` call fails":

```python
from injector.schedulers import GraphFaultScheduler

graph = GraphFaultScheduler()
graph.register_fault("init_transfer", TimeoutFault(delay_seconds=0.0), trigger_on="get_db_creds")
```

Trigger tools are observed but never faulted; each trigger call arms the target, the fault fires once,
then the target recovers until re-armed. Re-calling the trigger re-arms it - that is the circular-loop trap.

---

## What we measure

| Metric | Description |
| --- | --- |
| **Step Efficiency (clean)** | `clean_optimal / actual`. Penalizes every extra step vs. a no-fault world (blast radius / degradation). |
| **Step Efficiency (fault-aware)** | `(clean_optimal + scheduler.forced_extra_steps()) / actual`. Grades recovery quality given the scheduled faults were unavoidable; a perfect recovery scores 1.0. |
| **Self-Correction Recovery Rate** | Recovered states after a tool error, no human intervention. |
| **Graceful Termination** | Whether the run ends via a designated terminal path rather than an error. |
| **Cost/Token Blast Radius** | Cost variance and token bloat under degraded states (roadmap). |

`forced_extra_steps()` lives on the scheduler: the base returns 0; `GraphFaultScheduler` returns `2 x` armed targets (one unavoidable error + one recovery reasoning step).

---

## Observability

Every watched tool call is observed as an OpenInference-tagged OTel span
(`openinference.span.kind=TOOL`, `tool.name`, plus `fault.injected` and, when
applicable, `fault.type` / `error.type`). Spans go to:

- an **in-memory exporter** (source of truth for metrics)
- an optional **OTLP gRPC exporter** (`localhost:4317`) for a UI

Model-call tracing is opt-in via the framework tracing integration. Attach
once after creating `Instrumentation` and detach after the run:

```python
from telemetry.integrations.langchain import LangChainTracingIntegration

langchain_tracing = LangChainTracingIntegration()
langchain_tracing.attach(instrumentation)
try:
    runner.invoke(initial_state, thread_id="chaos-001")
finally:
    langchain_tracing.detach()
```

This uses the same tracer provider as the agent and tool spans. LangChain's
OpenInference instrumentor emits `LLM` spans; export normalization aliases
observed `llm.model_name` and token counts to `gen_ai.request.model` and
`gen_ai.usage.*`. Missing model metadata or usage stays missing; no token
counts are guessed. Count each framework tool invocation once; the run-ID bridge
enriches the existing LangChain TOOL span instead of exporting a sibling. Message
content may contain sensitive data and should be redacted before production use.

**Arize Phoenix** runs alongside the collector in Docker Compose:

```powershell
docker compose -f infrastructure/docker-compose.yaml up -d
docker compose -f infrastructure/docker-compose.yaml ps
```

Open `http://localhost:6006`. The application sends OTLP gRPC to
`localhost:4317` (the collector), which forwards spans to Phoenix over the
Docker network. The collector also logs spans to its `debug` exporter.
Phoenix persists SQLite data in the `phoenix_data` named volume. Do not run
`.venv\Scripts\python.exe -m phoenix.server.main serve` concurrently: the
local server would compete with the collector for host port 4317.

---

## Design guarantees

- **Deterministic.** The same fault schedule always produces the same trajectory - reproducible benchmarks, not random chaos.
- **Clone-don\'t-mutate.** Poisoning clones tools (`model_copy` + `func` swap); originals stay pristine and reusable, and no graph is ever mutated at runtime.
- **Strict ownership.** Two schedulers cannot target the same tool; the injector raises on conflict.
- **Dumb runners.** Runners receive a finished workflow and cannot leak faults between runs.
- **Pure catalog.** Workflow modules never import injector / runner / frameworks / telemetry.

---

## Project layout

```
injector/            chaos core: Fault, FaultInjector (dispatcher), ChaosToolProxy
  schedulers/          pluggable strategies: FixedCallScheduler, GraphFaultScheduler
  adapters/base.py     ToolAdapter ABC (framework-agnostic tool seam)
  faults/              TimeoutFault, RateLimitFault, MalformedJSONFault, SchemaMutationFault
frameworks/          Abstract Factory per framework
  base.py              FrameworkProvider ABC
  langgraph/           LangGraphProvider: runner + evaluator + tool adapter
runner/base.py       AgentRunner ABC (framework-agnostic executor)
telemetry/
  instrumentation.py   OTel TracerProvider + in-memory + OTLP exporters
  metrics/             Metric ABC; StepEfficiencyMetric (clean + fault-aware)
  evaluator.py         legacy Evaluator (recovery / graceful termination)
examples/
  langgraph/           pure workflow catalog
    minimal_llm_agent.py
    secure_migration_agent.py
  run_benchmark.py
  run_secure_migration_benchmark.py
infrastructure/      docker-compose for Phoenix + OTLP collector (optional)
```

---

## Roadmap

- Cost / token blast-radius metric (consume LLM spans from OpenInference auto-instrumentation).
- HTTP-level fault injection at the MCP transport layer (504s, connection drops, 429s).
- Async tool support, richer scheduler conditions, fault-reactive schedulers.
- Enterprise API & mocking engine; interactive telemetry dashboard.
