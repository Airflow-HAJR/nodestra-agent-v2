"""Turn-level latency accumulator. Reset once per invoke(), read after."""

_stats: dict = {
    "llm_ms": 0.0, "llm_calls": 0,
    "tool_ms": 0.0, "tool_calls": 0,
    "supermemory_ms": 0.0, "supermemory_calls": 0,
    "moss_ms": 0.0, "moss_calls": 0,
}


def reset():
    _stats.update({
        "llm_ms": 0.0, "llm_calls": 0,
        "tool_ms": 0.0, "tool_calls": 0,
        "supermemory_ms": 0.0, "supermemory_calls": 0,
        "moss_ms": 0.0, "moss_calls": 0,
    })
    import agent.graph as _graph
    _graph.reset_turn_state()


def add_llm(seconds: float):
    _stats["llm_ms"] += seconds * 1000
    _stats["llm_calls"] += 1


def add_tool(seconds: float):
    _stats["tool_ms"] += seconds * 1000
    _stats["tool_calls"] += 1


def add_supermemory(seconds: float):
    _stats["supermemory_ms"] += seconds * 1000
    _stats["supermemory_calls"] += 1


def add_moss(seconds: float):
    _stats["moss_ms"] += seconds * 1000
    _stats["moss_calls"] += 1


def _avg(ms: float, calls: int) -> str:
    return f"{ms / calls:.0f}ms avg" if calls else "—"


def summary(total_seconds: float) -> str:
    total_ms = total_seconds * 1000
    llm_ms = _stats["llm_ms"]
    tool_ms = _stats["tool_ms"]
    sm_ms = _stats["supermemory_ms"]
    sm_calls = _stats["supermemory_calls"]
    moss_ms = _stats["moss_ms"]
    moss_calls = _stats["moss_calls"]
    other_ms = total_ms - llm_ms - tool_ms
    return (
        f"  ⏱  total {total_ms:.0f}ms  |  "
        f"llm {llm_ms:.0f}ms × {_stats['llm_calls']}  |  "
        f"tools {tool_ms:.0f}ms × {_stats['tool_calls']}  |  "
        f"supermemory {_avg(sm_ms, sm_calls)} × {sm_calls}  |  "
        f"moss {_avg(moss_ms, moss_calls)} × {moss_calls}  |  "
        f"other {other_ms:.0f}ms"
    )
