"""Turn-level latency accumulator. Reset once per invoke(), read after."""

_stats: dict = {
    "llm_ms": 0.0, "llm_calls": 0,
    "tool_ms": 0.0, "tool_calls": 0,
    "tts_ms": 0.0, "tts_calls": 0,
}


def reset():
    _stats.update({
        "llm_ms": 0.0, "llm_calls": 0,
        "tool_ms": 0.0, "tool_calls": 0,
        "tts_ms": 0.0, "tts_calls": 0,
    })
    import agent.graph as _graph
    _graph.reset_turn_state()


def add_llm(seconds: float):
    _stats["llm_ms"] += seconds * 1000
    _stats["llm_calls"] += 1


def add_tool(seconds: float):
    _stats["tool_ms"] += seconds * 1000
    _stats["tool_calls"] += 1


def add_tts(seconds: float):
    _stats["tts_ms"] += seconds * 1000
    _stats["tts_calls"] += 1


def _fmt(ms: float, calls: int) -> str:
    return f"{ms:.0f}ms×{calls}" if calls else "—"


def summary(total_seconds: float) -> str:
    total_ms = total_seconds * 1000
    llm_ms   = _stats["llm_ms"]
    tool_ms  = _stats["tool_ms"]
    tts_ms   = _stats["tts_ms"]
    other_ms = total_ms - llm_ms - tool_ms - tts_ms
    return (
        f"LATENCY  total={total_ms:.0f}ms"
        f"  llm={_fmt(llm_ms, _stats['llm_calls'])}"
        f"  tools={_fmt(tool_ms, _stats['tool_calls'])}"
        f"  tts={_fmt(tts_ms, _stats['tts_calls'])}"
        f"  other={other_ms:.0f}ms"
    )
