import hashlib
import hmac
import json
import logging
import queue
import threading
import time
from typing import Any, Iterator

from agentphone import AgentPhone
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel

from agent.config import AGENTPHONE_API_KEY, AGENTPHONE_WEBHOOK_SECRET
from agent.graph import ITERATION_CAP, bind_speak_early_callback, graph
from agent import timing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lincoln Airport Agent")

# ── Request deduplication ─────────────────────────────────────────────────────

_dedup_lock = threading.Lock()
_recent_requests: dict[str, float] = {}
_DEDUP_TTL = 30.0  # seconds


def _check_and_register(call_id: str, user_text: str) -> bool:
    """Return True if this is a new request and should be processed; False if duplicate."""
    key = f"{call_id}:{hashlib.md5(user_text.encode()).hexdigest()}"
    now = time.time()
    with _dedup_lock:
        expired = [k for k, t in _recent_requests.items() if now - t > _DEDUP_TTL]
        for k in expired:
            del _recent_requests[k]
        if key in _recent_requests:
            return False
        _recent_requests[key] = now
        return True


_agentphone = AgentPhone(api_key=AGENTPHONE_API_KEY) if AGENTPHONE_API_KEY else None
_agent_id = "cmpa2b618030ojz00wyj247l8"
_number_id = "cmp7k8i5403so807bvhzljclg"
DEFAULT_EARLY_SPEAK_TEXT = "Let me check that for you."
STREAM_QUEUE_SENTINEL = object()
EARLY_SPEAK_FALLBACK_SECONDS = 0.75


# ── Signature verification ────────────────────────────────────────────────────

def _verify_signature(payload_body: bytes, signature: str, secret: str, timestamp: str) -> bool:
    # AgentPhone signs "{timestamp}.{raw_body}"
    signed = f"{timestamp}.".encode() + payload_body
    expected = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    result = hmac.compare_digest(f"sha256={expected}", signature)

    if not result:
        logger.warning(f"Sig mismatch — received: {signature!r}  expected: sha256={expected!r}")

    return result


# ── Outbound notifications ────────────────────────────────────────────────────

class GateChangeRequest(BaseModel):
    phone: str
    flight: str
    old_gate: str
    new_gate: str


def _notify_via_agentphone(phone: str, message: str, method: str = "call"):
    if not _agentphone:
        return None

    try:
        if method == "call":
            return _agentphone.calls._post("/v1/calls", to=phone, message=message)
        else:
            return _agentphone.messages.send(
                agent_id=_agent_id,
                to_number=phone,
                body=message,
                number_id=_number_id,
            )
    except Exception as e:
        logger.error(f"AgentPhone {method} failed: {e}")
        return None


def _build_graph_config(thread_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": ITERATION_CAP,
    }


def _extract_response_text(result: dict[str, Any]) -> str | None:
    for message in reversed(result.get("messages", [])):
        if hasattr(message, "content") and message.content and not getattr(message, "tool_calls", None):
            return message.content if isinstance(message.content, str) else str(message.content)
    return None


def _graph_reply(user_text: str, thread_id: str, user_id: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"messages": [HumanMessage(content=user_text)]}
    if user_id:
        payload["user_id"] = user_id

    result = graph.invoke(payload, config=_build_graph_config(thread_id))

    response_text = _extract_response_text(result)
    if not response_text:
        return {"text": "I'm having trouble with that. Please try again."}

    logger.info(f"{thread_id} response: {response_text[:100]}")
    reply = {"text": response_text}
    if result.get("should_end", False):
        reply["hangup"] = True
    return reply


def _voice_webhook_stream(call_id: str, user_text: str, user_id: str | None = None) -> Iterator[bytes]:
    payloads: queue.Queue[dict[str, Any] | object] = queue.Queue()
    interim_sent = threading.Event()

    def emit(payload: dict[str, Any]) -> None:
        payloads.put(payload)

    def emit_interim(text: str) -> None:
        if interim_sent.is_set():
            return
        interim_sent.set()
        emit({"text": text, "interim": True})

    def worker() -> None:
        try:
            timing.reset()
            with bind_speak_early_callback(emit_interim):
                emit(_graph_reply(user_text, call_id, user_id))
        except GraphRecursionError:
            emit({"text": "I'm getting a bit turned around. Could you re-state what you need?"})
        except Exception as e:
            logger.error(f"Graph failed for {call_id}: {e}", exc_info=True)
            emit({"text": "Something went wrong. Please try again or ask airport staff for help."})
        finally:
            payloads.put(STREAM_QUEUE_SENTINEL)

    threading.Thread(target=worker, name=f"agentphone-{call_id}", daemon=True).start()

    try:
        first_payload = payloads.get(timeout=EARLY_SPEAK_FALLBACK_SECONDS)
    except queue.Empty:
        emit_interim(DEFAULT_EARLY_SPEAK_TEXT)
        first_payload = payloads.get()

    if first_payload is not STREAM_QUEUE_SENTINEL:
        yield (json.dumps(first_payload) + "\n").encode()

    while True:
        payload = payloads.get()
        if payload is STREAM_QUEUE_SENTINEL:
            return
        yield (json.dumps(payload) + "\n").encode()


# ── Webhook ───────────────────────────────────────────────────────────────────

@app.post("/")
@app.post("/webhook")
async def webhook(request: Request):
    """Handle incoming AgentPhone calls and messages."""
    raw_body = await request.body()

    # Verify signature when secret is configured
    if AGENTPHONE_WEBHOOK_SECRET:
        signature = request.headers.get("X-Webhook-Signature", "")
        timestamp = request.headers.get("X-Webhook-Timestamp", "")
        if not _verify_signature(raw_body, signature, AGENTPHONE_WEBHOOK_SECRET, timestamp):
            logger.warning("Webhook signature verification failed")
            raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        body = json.loads(raw_body)
    except Exception as e:
        logger.error(f"Failed to parse webhook body: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event = body.get("event")
    channel = body.get("channel")
    data = body.get("data", {})

    logger.info(f"Webhook: event={event}, channel={channel}")

    # Ignore non-message events (call_ended, status updates, etc.)
    if event != "agent.message":
        return {"status": "ok"}

    # Use conversationId if present, fall back to caller's phone number for SMS threads
    call_id = (
        data.get("callId")
        or (data.get("conversationId") or None)
        or data.get("from")
        or body.get("agentId")
    )
    if not call_id:
        logger.warning("Missing call ID in payload")
        raise HTTPException(status_code=400, detail="Missing call ID")

    # Caller's phone number is the stable user identity across sessions
    user_id = data.get("from") or None

    # Voice uses transcript (string); SMS uses message
    if channel == "voice":
        transcript = data.get("transcript", "")
        user_text = (transcript if isinstance(transcript, str) else "").strip()
    else:
        user_text = data.get("message", "").strip()

    if not user_text:
        logger.warning(f"Empty message for {call_id}")
        return {"text": "I didn't catch that. Can you repeat?"}

    if not _check_and_register(call_id, user_text):
        logger.info(f"Duplicate request for {call_id}, skipping inference")
        return {"status": "ok"}

    logger.info(f"{call_id}: '{user_text}'")

    if channel == "voice":
        return StreamingResponse(
            _voice_webhook_stream(call_id, user_text, user_id),
            media_type="application/x-ndjson",
        )

    timing.reset()

    try:
        reply = _graph_reply(user_text, call_id, user_id)
    except GraphRecursionError:
        reply = {"text": "I'm getting a bit turned around. Could you re-state what you need?"}
    except Exception as e:
        logger.error(f"Graph failed for {call_id}: {e}", exc_info=True)
        reply = {"text": "Something went wrong. Please try again or ask airport staff for help."}

    reply_text = reply.get("text", "")
    if reply_text and user_id:
        _notify_via_agentphone(user_id, reply_text, method="imessage")

    return {"status": "ok"}


# ── Gate change notifications ─────────────────────────────────────────────────

@app.post("/gate-change")
async def gate_change(body: GateChangeRequest):
    """Notify a passenger of a gate change via outbound call or SMS."""
    message = (
        f"Hi, this is Oakland Airport. Your flight {body.flight} "
        f"gate has changed from {body.old_gate} to {body.new_gate}. "
        f"Please proceed to gate {body.new_gate}. Thank you."
    )

    if not AGENTPHONE_API_KEY:
        logger.info(f"[MOCK] → {body.phone}: {message}")
        return {
            "status": "mock_notified",
            "message": message,
            "note": "Set AGENTPHONE_API_KEY to send real notifications",
        }

    result = _notify_via_agentphone(body.phone, message, method="call")
    if result:
        logger.info(f"Call placed to {body.phone}")
        return {"status": "called", "result": result}

    # Fall back to SMS
    result = _notify_via_agentphone(body.phone, message, method="imessage")
    if result:
        logger.info(f"SMS sent to {body.phone}")
        return {"status": "sms_sent", "result": result}

    return {"status": "error", "error": "Failed to notify via call or SMS"}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "Lincoln Airport Agent"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
