"""
Minimal text chat UI for testing the agent + memory system — no voice, no Twilio.

Run:
    uv run uvicorn webchat:app --reload --port 8100
Then open http://localhost:8100

Left panel is a live view of what the agent has remembered about the current
user id. Change the user id to simulate a returning caller and watch stored
memories load at the start of the session.
"""
from __future__ import annotations

import time
import uuid

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent.config import FAST_LLM_LABEL, OPENAI_MODEL_MAIN
from agent.graph import ITERATION_CAP, get_turn_tools_used, graph, save_memory_in_background
from agent.vector_memory import fetch_all_memories
from agent import timing

app = FastAPI(title="Agent Memory Test UI")

# session_id -> thread_id, so each browser tab keeps one LangGraph checkpoint.
_threads: dict[str, str] = {}


def _thread_for(session_id: str) -> str:
    if session_id not in _threads:
        _threads[session_id] = str(uuid.uuid4())
    return _threads[session_id]


class ChatIn(BaseModel):
    user_id: str | None = None
    session_id: str
    message: str


class ResetIn(BaseModel):
    session_id: str


def _memory_view(user_id: str | None) -> list[dict]:
    if not user_id:
        return []
    rows = fetch_all_memories(user_id)
    return [
        {"content": r.get("content"), "category": r.get("category")}
        for r in rows
    ]


@app.post("/api/chat")
def chat(body: ChatIn):
    thread_id = _thread_for(body.session_id)
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": ITERATION_CAP,
    }
    payload: dict = {"messages": [HumanMessage(content=body.message)]}
    if body.user_id:
        payload["user_id"] = body.user_id
        # This harness exists to exercise durable memory, so a user id typed
        # into it stands in for a signed-in account rather than a guest.
        payload["persist_memory"] = True

    timing.reset()
    t0 = time.time()
    result = graph.invoke(payload, config)  # type: ignore
    turn_secs = time.time() - t0
    snap = timing.turn_snapshot(turn_secs)
    # Memory extraction runs in the background — never blocks the reply.
    save_memory_in_background(config, result)

    reply = ""
    for m in reversed(result.get("messages", [])):
        if getattr(m, "content", None) and not getattr(m, "tool_calls", None):
            reply = m.content if isinstance(m.content, str) else str(m.content)
            break

    # Reflect the in-session cache (includes anything saved this turn).
    cached = result.get("user_memories") or []
    memories = [{"content": r.get("content"), "category": r.get("category")} for r in cached]

    latency = {
        "total_ms": round(snap["total_ms"]),
        "main_llm_ms": round(snap["main_llm_ms"]),
        "main_llm_calls": snap["main_llm_calls"],
        "main_llm_model": OPENAI_MODEL_MAIN,
        "fast_llm_ms": round(snap["fast_llm_ms"]),
        "fast_llm_calls": snap["fast_llm_calls"],
        "fast_llm_model": FAST_LLM_LABEL,
        "tool_ms": round(snap["tool_ms"]),
        "tool_calls": snap["tool_calls"],
        "tools_used": get_turn_tools_used(),
        "memory_ms": round(snap["memory_ms"]),
        "other_ms": round(snap["other_ms"]),
    }
    timing.reset()

    return {"reply": reply, "memories": memories, "latency": latency}


@app.get("/api/memories")
def memories(user_id: str | None = None):
    return {"memories": _memory_view(user_id)}


@app.post("/api/reset")
def reset(body: ResetIn):
    _threads[body.session_id] = str(uuid.uuid4())
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    return _HTML


_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Agent Memory Test</title>
<style>
  :root {
    --bg:#0f1115; --panel:#171a21; --border:#262b36; --text:#e6e9ef;
    --muted:#8b93a5; --accent:#5b9dff; --user:#264a7a; --bot:#20242e;
  }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         background:var(--bg); color:var(--text); height:100vh; display:flex; }
  .sidebar { width:320px; border-right:1px solid var(--border); background:var(--panel);
             display:flex; flex-direction:column; padding:16px; gap:14px; }
  .sidebar h2 { margin:0; font-size:13px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); }
  label { display:block; font-size:12px; color:var(--muted); margin-bottom:4px; }
  input { width:100%; padding:8px 10px; background:var(--bg); border:1px solid var(--border);
          border-radius:8px; color:var(--text); font-size:14px; }
  button { padding:8px 12px; background:var(--accent); color:#fff; border:none; border-radius:8px;
           cursor:pointer; font-size:13px; font-weight:600; }
  button.ghost { background:transparent; color:var(--muted); border:1px solid var(--border); }
  .mem-list { flex:1; overflow-y:auto; display:flex; flex-direction:column; gap:8px; }
  .mem { background:var(--bg); border:1px solid var(--border); border-radius:8px; padding:8px 10px; }
  .mem .cat { font-size:10px; text-transform:uppercase; letter-spacing:.06em; color:var(--accent); }
  .mem .txt { margin-top:2px; }
  .empty { color:var(--muted); font-style:italic; padding:8px 0; }
  .main { flex:1; display:flex; flex-direction:column; }
  .transcript { flex:1; overflow-y:auto; padding:24px; display:flex; flex-direction:column; gap:12px; }
  .msg { max-width:70%; padding:10px 14px; border-radius:14px; white-space:pre-wrap; }
  .msg.user { align-self:flex-end; background:var(--user); border-bottom-right-radius:4px; }
  .msg.bot  { align-self:flex-start; background:var(--bot); border:1px solid var(--border); border-bottom-left-radius:4px; }
  .msg.sys  { align-self:center; color:var(--muted); font-size:12px; font-style:italic; background:none; }
  .composer { display:flex; gap:10px; padding:16px 24px; border-top:1px solid var(--border); background:var(--panel); }
  .composer input { flex:1; }
  .dot { display:inline-block; width:6px; height:6px; border-radius:50%; background:var(--muted);
         animation:blink 1.2s infinite; margin:0 1px; }
  .dot:nth-child(2){animation-delay:.2s} .dot:nth-child(3){animation-delay:.4s}
  @keyframes blink{0%,60%,100%{opacity:.3}30%{opacity:1}}
  .latency { font-size:11px; color:var(--muted); margin-top:6px; }
  .latency summary { cursor:pointer; list-style:none; }
  .latency summary::-webkit-details-marker { display:none; }
  .latency summary::before { content:"⏱ "; }
  .latency .rows { margin-top:4px; display:flex; flex-direction:column; gap:2px; padding-left:4px; }
  .latency .row { display:flex; justify-content:space-between; gap:12px; }
  .latency .label { color:var(--muted); }
  .latency .val { color:var(--text); font-variant-numeric:tabular-nums; }
</style>
</head>
<body>
  <div class="sidebar">
    <h2>Session</h2>
    <div>
      <label>User ID (phone / any string)</label>
      <input id="userId" value="+15551234567"/>
    </div>
    <div style="display:flex; gap:8px;">
      <button id="newConvo" class="ghost" style="flex:1">New conversation</button>
    </div>
    <h2>Remembered about this user</h2>
    <div class="mem-list" id="memList"><div class="empty">Loading…</div></div>
  </div>
  <div class="main">
    <div class="transcript" id="transcript"></div>
    <div class="composer">
      <input id="input" placeholder="Type a message…" autocomplete="off"/>
      <button id="send">Send</button>
    </div>
  </div>

<script>
const sessionId = crypto.randomUUID();
const $ = s => document.querySelector(s);
const transcript = $("#transcript");

function addMsg(role, text, latency){
  const wrap = document.createElement("div");
  wrap.style.cssText = "display:flex;flex-direction:column;" + (role==="user" ? "align-items:flex-end" : "align-items:flex-start");
  const el = document.createElement("div");
  el.className = "msg " + role;
  el.textContent = text;
  wrap.appendChild(el);
  if(latency && role === "bot"){
    const tools = latency.tools_used && latency.tools_used.length
      ? latency.tools_used.join(", ")
      : "—";
    const rows = [
      ["Main LLM", `${latency.main_llm_model}: ${latency.main_llm_ms}ms`],
      ["Fast LLM", `${latency.fast_llm_model}: ${latency.fast_llm_ms}ms ×${latency.fast_llm_calls}`],
      ["Tools", `${tools}: ${latency.tool_ms}ms`],
      ["Memory DB", `${latency.memory_ms}ms`],
      ["Network/other", `${latency.other_ms}ms`],
    ];
    const det = document.createElement("details");
    det.className = "latency";
    const sum = document.createElement("summary");
    sum.textContent = `${latency.total_ms}ms`;
    det.appendChild(sum);
    const rowsEl = document.createElement("div");
    rowsEl.className = "rows";
    for(const [label, val] of rows){
      const row = document.createElement("div");
      row.className = "row";
      row.innerHTML = `<span class="label">${label}</span><span class="val">${val}</span>`;
      rowsEl.appendChild(row);
    }
    det.appendChild(rowsEl);
    wrap.appendChild(det);
  }
  transcript.appendChild(wrap);
  transcript.scrollTop = transcript.scrollHeight;
  return el;
}

function renderMemories(mems){
  const list = $("#memList");
  list.innerHTML = "";
  if(!mems || !mems.length){ list.innerHTML = '<div class="empty">Nothing remembered yet.</div>'; return; }
  for(const m of mems){
    const el = document.createElement("div");
    el.className = "mem";
    el.innerHTML = `<div class="cat">${m.category||"other"}</div><div class="txt"></div>`;
    el.querySelector(".txt").textContent = m.content || "";
    list.appendChild(el);
  }
}

async function loadMemories(){
  const uid = encodeURIComponent($("#userId").value.trim());
  const r = await fetch("/api/memories?user_id=" + uid);
  const d = await r.json();
  renderMemories(d.memories);
}

async function send(){
  const input = $("#input");
  const text = input.value.trim();
  if(!text) return;
  input.value = "";
  addMsg("user", text);
  const typing = addMsg("bot", "");
  typing.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
  // Wrap the bot bubble in a column container so latency can be appended below it.
  if(typing.parentElement.style.cssText === ""){
    typing.parentElement.style.cssText = "display:flex;flex-direction:column;align-items:flex-start";
  }
  try{
    const r = await fetch("/api/chat", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ user_id: $("#userId").value.trim(), session_id: sessionId, message: text })
    });
    const d = await r.json();
    // Replace the typing indicator's parent wrapper with the real reply+latency
    const wrap = typing.parentElement;
    typing.textContent = d.reply || "(no reply)";
    if(d.latency && wrap){
      const tools = d.latency.tools_used && d.latency.tools_used.length
        ? d.latency.tools_used.join(", ")
        : "—";
      const rows = [
        ["Main LLM", `${d.latency.main_llm_model}: ${d.latency.main_llm_ms}ms`],
        ["Fast LLM", `${d.latency.fast_llm_model}: ${d.latency.fast_llm_ms}ms ×${d.latency.fast_llm_calls}`],
        ["Tools", `${tools}: ${d.latency.tool_ms}ms`],
        ["Memory DB", `${d.latency.memory_ms}ms`],
        ["Network/other", `${d.latency.other_ms}ms`],
      ];
      const det = document.createElement("details");
      det.className = "latency";
      const sum = document.createElement("summary");
      sum.textContent = `${d.latency.total_ms}ms`;
      det.appendChild(sum);
      const rowsEl = document.createElement("div");
      rowsEl.className = "rows";
      for(const [label, val] of rows){
        const row = document.createElement("div");
        row.className = "row";
        row.innerHTML = `<span class="label">${label}</span><span class="val">${val}</span>`;
        rowsEl.appendChild(row);
      }
      det.appendChild(rowsEl);
      wrap.appendChild(det);
    }
    renderMemories(d.memories);
    // Memory extraction runs in the background AFTER the reply, so a fact the
    // user just mentioned isn't saved yet when this response returns. Re-poll
    // the panel a few times so it appears once the background save lands.
    setTimeout(loadMemories, 1500);
    setTimeout(loadMemories, 3500);
    setTimeout(loadMemories, 6000);
  }catch(e){
    typing.className = "msg sys";
    typing.textContent = "Error: " + e;
  }
}

$("#send").onclick = send;
$("#input").addEventListener("keydown", e => { if(e.key==="Enter") send(); });
$("#userId").addEventListener("change", () => { transcript.innerHTML=""; loadMemories(); });
$("#newConvo").onclick = async () => {
  await fetch("/api/reset", { method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({ session_id: sessionId }) });
  transcript.innerHTML = "";
  addMsg("sys", "Started a new conversation. Stored memories persist for this user.");
  loadMemories();
};

loadMemories();
addMsg("sys", "Say something like: \\"I only eat halal food\\" or \\"I fly Southwest\\", then start a new conversation and ask for recommendations.");
</script>
</body>
</html>
"""
