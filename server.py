import hashlib
import hmac
import httpx
import logging

from fastapi import FastAPI, HTTPException, Request
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent.config import AGENTPHONE_API_KEY, AGENTPHONE_WEBHOOK_SECRET
from agent.graph import graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lincoln Airport Agent")
AGENTPHONE_BASE_URL = "https://api.agentphone.ai/v1"


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
    if not AGENTPHONE_API_KEY:
        return None

    headers = {
        "Authorization": f"Bearer {AGENTPHONE_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        if method == "call":
            url = f"{AGENTPHONE_BASE_URL}/calls"
            payload = {"to": phone, "message": message}
        else:
            url = f"{AGENTPHONE_BASE_URL}/sms"
            payload = {"to": phone, "body": message}

        response = httpx.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"AgentPhone {method} failed: {e}")
        return None


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
        body = request.app.state  # parse below
        import json
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

    # Voice uses transcript (string); SMS uses message
    if channel == "voice":
        transcript = data.get("transcript", "")
        user_text = (transcript if isinstance(transcript, str) else "").strip()
    else:
        user_text = data.get("message", "").strip()

    if not user_text:
        logger.warning(f"Empty message for {call_id}")
        return {"text": "I didn't catch that. Can you repeat?"}

    logger.info(f"{call_id}: '{user_text}'")

    # Run through LangGraph — callId is thread_id for multi-turn continuity
    try:
        config = {"configurable": {"thread_id": call_id}}
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config=config,
        )

        messages = result.get("messages", [])
        if not messages:
            return {"text": "I'm having trouble with that. Please try again."}

        last_msg = messages[-1]
        response_text = (
            last_msg.content
            if isinstance(last_msg.content, str)
            else str(last_msg.content)
        )

        logger.info(f"{call_id} response: {response_text[:100]}")

        # Tell AgentPhone to hang up when conversation is complete
        should_end = result.get("should_end", False)
        if channel == "voice" and should_end:
            return {"text": response_text, "hangup": True}

        return {"text": response_text}

    except Exception as e:
        logger.error(f"Graph failed for {call_id}: {e}", exc_info=True)
        return {"text": "Something went wrong. Please try again or ask airport staff for help."}


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
    result = _notify_via_agentphone(body.phone, message, method="sms")
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
