"""Turn-level latency accumulator. Reset once per invoke(), read after."""

_stats: dict = {"llm_ms": 0.0, "tool_ms": 0.0, "llm_calls": 0, "tool_calls": 0}


def reset():
    _stats.update({"llm_ms": 0.0, "tool_ms": 0.0, "llm_calls": 0, "tool_calls": 0})
    import agent.graph as _graph
    _graph.reset_turn_state()


def add_llm(seconds: float):
    _stats["llm_ms"] += seconds * 1000
    _stats["llm_calls"] += 1


def add_tool(seconds: float):
    _stats["tool_ms"] += seconds * 1000
    _stats["tool_calls"] += 1


def summary(total_seconds: float) -> str:
    total_ms = total_seconds * 1000
    llm_ms = _stats["llm_ms"]
    tool_ms = _stats["tool_ms"]
    other_ms = total_ms - llm_ms - tool_ms
    return (
        f"  ⏱  total {total_ms:.0f}ms  |  "
        f"llm {llm_ms:.0f}ms × {_stats['llm_calls']}  |  "
        f"tools {tool_ms:.0f}ms × {_stats['tool_calls']}  |  "
        f"other {other_ms:.0f}ms"
    )
