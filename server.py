import asyncio
import base64
import contextvars
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import parse_qs

import requests as _requests

from deepgram import AsyncDeepgramClient
from deepgram.listen.v2.types.listen_v2turn_info import ListenV2TurnInfo
from elevenlabs import ElevenLabs
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel
from twilio.request_validator import RequestValidator
from twilio.rest import Client as TwilioClient
from twilio.twiml.messaging_response import MessagingResponse
from twilio.twiml.voice_response import Connect, Gather, Stream, VoiceResponse

from agent.analytics import finish_call, hash_user_id, insert_call, insert_turn, start_call
from agent.graph import bind_map_callback, bind_sentence_callback, bind_speak_early_callback, bind_tool_status_callback, get_turn_tools_used
from agent.config import (
    ALLOWED_ORIGINS,
    CARTESIA_API_KEY,
    CARTESIA_EMOTION,
    CARTESIA_MODEL_ID,
    CARTESIA_VOICE_ID,
    DEFAULT_AIRPORT,
    DEEPGRAM_API_KEY,
    ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID,
    GATEGETTER_URL,
    SERVER_BASE_URL,
    TTS_PROVIDER,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_PHONE_NUMBER,
)
from agent.db import invalidate_map_cache, load_map_levels
from agent.graph import ITERATION_CAP, graph
from agent.summarizer import summarize
from agent import timing
from agent.logger import log_event

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Short present-tense labels shown in the web UI's status bar while a tool runs,
# keyed by tool name (see agent.graph._TOOL_SPEAK_PRIORITY for the matching
# TTS phrases used on the phone line — kept separate since these are UI-only).
_TOOL_STATUS_LABELS: dict[str, str] = {
    "get_route": "Charting course...",
    "find_poi": "Searching map...",
    "find_nearest": "Finding nearest option...",
    "resolve_poi": "Narrowing it down...",
    "get_nodes": "Loading the map...",
    "recall_user_memories": "Checking your preferences...",
    "search_flight_info": "Checking flights...",
    "get_flight_status": "Checking flight status...",
    "search_store_info": "Searching shops...",
    "track_flight_changes": "Setting up flight tracking...",
    "update_user_memory": "Remembering that...",
    "search_user_memories": "Checking your preferences...",
}

# Suppress noisy third-party HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)

app = FastAPI(title="Oakland Airport Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _origin_allowed(origin: str | None) -> bool:
    """CORSMiddleware only guards HTTP requests, not the WebSocket handshake —
    browsers don't apply same-origin policy to WS the way they do to fetch/XHR.
    Callers of /web/stream must check the Origin header themselves."""
    return "*" in ALLOWED_ORIGINS or origin in ALLOWED_ORIGINS

_executor = ThreadPoolExecutor(max_workers=10)

# mulaw silence ≈ 0xFF/0x7F; voice = significant deviation from those values.
_MULAW_SILENCE = frozenset({0xFF, 0x7F})
_VAD_THRESHOLD = 0.10  # fraction of non-silence bytes required to consider chunk voiced

def _mulaw_voice_ratio(data: bytes) -> float:
    if not data:
        return 0.0
    return sum(1 for b in data if b not in _MULAW_SILENCE) / len(data)

def _mulaw_has_voice(data: bytes) -> bool:
    """Return True if mulaw audio chunk contains voice activity."""
    return len(data) >= 20 and _mulaw_voice_ratio(data) > _VAD_THRESHOLD

# ── Clients ───────────────────────────────────────────────────────────────────

_twilio = (
    TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN
    else None
)
_validator = RequestValidator(TWILIO_AUTH_TOKEN) if TWILIO_AUTH_TOKEN else None
_eleven = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None

# ── ElevenLabs TTS ────────────────────────────────────────────────────────────

_FALLBACK_VOICE = "Polly.Joanna"

# MP3 cache for <Play> URLs (welcome message, outbound calls)
_audio_lock = threading.Lock()
_audio_cache: dict[str, tuple[bytes, float]] = {}
_AUDIO_TTL = 120.0


def _store_audio(audio_bytes: bytes) -> str:
    token = str(uuid.uuid4())
    now = time.time()
    with _audio_lock:
        stale = [k for k, (_, t) in _audio_cache.items() if now - t > _AUDIO_TTL]
        for k in stale:
            del _audio_cache[k]
        _audio_cache[token] = (audio_bytes, now)
    return token


@app.get("/audio/{token}")
async def serve_audio(token: str):
    """One-shot MP3 endpoint used by Twilio <Play> for the welcome message."""
    with _audio_lock:
        entry = _audio_cache.pop(token, None)
    if not entry:
        raise HTTPException(status_code=404, detail="Audio not found or already played")
    audio_bytes, _ = entry
    return Response(content=audio_bytes, media_type="audio/mpeg")


def _elevenlabs_tts_mp3_url(text: str) -> str | None:
    """Generate ElevenLabs MP3, cache it, return a /audio/{token} URL for TwiML <Play>."""
    if not _eleven or not ELEVENLABS_VOICE_ID:
        return None
    try:
        audio_bytes = b"".join(
            _eleven.text_to_speech.convert(
                voice_id=ELEVENLABS_VOICE_ID,
                text=text,
                model_id="eleven_flash_v2_5",
                output_format="mp3_44100_128",
            )
        )
        return f"{SERVER_BASE_URL}/audio/{_store_audio(audio_bytes)}"
    except Exception as e:
        logger.error(f"ElevenLabs TTS (mp3) failed: {e}")
        return None


def _elevenlabs_tts_mulaw_iter(text: str):
    """Yield ElevenLabs mulaw 8 kHz chunks as they are generated."""
    if not _eleven or not ELEVENLABS_VOICE_ID:
        return iter([])
    try:
        return _eleven.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text=text,
            model_id="eleven_flash_v2_5",
            output_format="ulaw_8000",
        )
    except Exception as e:
        logger.error(f"ElevenLabs TTS (mulaw) failed: {e}")
        return iter([])


# ── Cartesia TTS ──────────────────────────────────────────────────────────────

_CARTESIA_URL = "https://api.cartesia.ai/tts/bytes"
_CARTESIA_HEADERS = {
    "Cartesia-Version": "2026-03-01",
    "Content-Type": "application/json",
}


def _cartesia_tts_mp3_url(text: str) -> str | None:
    """Generate Cartesia MP3, cache it, return a /audio/{token} URL for TwiML <Play>."""
    if not CARTESIA_API_KEY:
        return None
    try:
        resp = _requests.post(
            _CARTESIA_URL,
            headers={**_CARTESIA_HEADERS, "X-API-Key": CARTESIA_API_KEY},
            json={
                "model_id": CARTESIA_MODEL_ID,
                "transcript": text,
                "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
                "output_format": {"container": "mp3", "encoding": "mp3", "sample_rate": 44100},
                "language": "en",
                "generation_config": {"emotion": CARTESIA_EMOTION},
            },
            timeout=10,
        )
        resp.raise_for_status()
        return f"{SERVER_BASE_URL}/audio/{_store_audio(resp.content)}"
    except Exception as e:
        logger.error(f"Cartesia TTS (mp3) failed: {e}")
        return None


def _cartesia_tts_mulaw_iter(text: str):
    """Fetch Cartesia mulaw 8 kHz audio and yield it as a single chunk."""
    if not CARTESIA_API_KEY:
        return iter([])
    try:
        resp = _requests.post(
            _CARTESIA_URL,
            headers={**_CARTESIA_HEADERS, "X-API-Key": CARTESIA_API_KEY},
            json={
                "model_id": CARTESIA_MODEL_ID,
                "transcript": text,
                "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
                "output_format": {"container": "raw", "encoding": "pcm_mulaw", "sample_rate": 8000},
                "language": "en",
                "generation_config": {"emotion": CARTESIA_EMOTION},
            },
            timeout=15,
        )
        resp.raise_for_status()
        return iter([resp.content])
    except Exception as e:
        logger.error(f"Cartesia TTS (mulaw) failed: {e}")
        return iter([])


# ── TTS dispatch ──────────────────────────────────────────────────────────────

def _tts_mp3_url(text: str) -> str | None:
    if TTS_PROVIDER == "cartesia":
        return _cartesia_tts_mp3_url(text)
    return _elevenlabs_tts_mp3_url(text)


def _tts_mulaw_iter(text: str):
    if TTS_PROVIDER == "cartesia":
        return _cartesia_tts_mulaw_iter(text)
    return _elevenlabs_tts_mulaw_iter(text)


# ── Request deduplication ─────────────────────────────────────────────────────

_dedup_lock = threading.Lock()
_recent_requests: dict[str, float] = {}
_DEDUP_TTL = 30.0

# SMS session tracking: from_number → {started_at, turn_count}
_sms_sessions: dict[str, dict] = {}


def _check_and_register(call_id: str, user_text: str) -> bool:
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


# ── Twilio helpers ────────────────────────────────────────────────────────────

def _verify_twilio_signature(request: Request, params: dict[str, str]) -> bool:
    if not _validator:
        logger.warning("TWILIO_AUTH_TOKEN not set — skipping signature verification")
        return True
    signature = request.headers.get("X-Twilio-Signature", "")

    def _candidates() -> list[str]:
        urls = []
        path = request.url.path
        query = f"?{request.url.query}" if request.url.query else ""

        # Raw URL as seen by the app (works when not behind a proxy).
        urls.append(str(request.url))

        # Reconstruct from forwarded headers (Railway / any reverse proxy).
        proto = request.headers.get("X-Forwarded-Proto", "")
        host = request.headers.get("X-Forwarded-Host") or request.headers.get("Host", "")
        if proto and host:
            urls.append(f"{proto}://{host}{path}{query}")

        # Explicit SERVER_BASE_URL override (local tunnels, etc.).
        base = (SERVER_BASE_URL or "").rstrip("/")
        if base:
            urls.append(f"{base}{path}{query}")

        return urls

    candidates = _candidates()
    logger.info(
        "Twilio sig check — token_prefix=%s sig=%s params=%s urls=%s",
        TWILIO_AUTH_TOKEN[:6] if TWILIO_AUTH_TOKEN else "NONE",
        signature[:10] if signature else "NONE",
        dict(list(params.items())[:3]),
        candidates,
    )
    for url in candidates:
        if _validator.validate(url, params, signature):
            return True

    logger.warning("Twilio signature validation failed for all candidate URLs")
    return False


def _parse_form(raw_body: bytes) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(raw_body.decode(), keep_blank_values=True).items()}


def _build_gather_twiml(prompt: str) -> VoiceResponse:
    """Build TwiML that prompts then listens using Twilio's built-in speech-to-text."""
    vr = VoiceResponse()
    gather = Gather(
        input="speech",
        speech_timeout="auto",
        action="/twilio/voice/process",
        method="POST",
    )
    gather.say(prompt, voice=_FALLBACK_VOICE)
    vr.append(gather)
    vr.say("I did not hear anything. Let's try again.", voice=_FALLBACK_VOICE)
    vr.redirect("/twilio/voice")
    return vr


# ── Agent graph helpers ───────────────────────────────────────────────────────

def _build_graph_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": ITERATION_CAP}


def _extract_response_text(result: dict[str, Any]) -> str | None:
    for message in reversed(result.get("messages", [])):
        if hasattr(message, "content") and message.content and not getattr(message, "tool_calls", None):
            return message.content if isinstance(message.content, str) else str(message.content)
    return None


def _graph_reply(
    user_text: str,
    thread_id: str,
    user_id: str | None = None,
    user_location: dict | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"messages": [HumanMessage(content=user_text)]}
    if user_id:
        payload["user_id"] = user_id
    if user_location:
        payload["user_location"] = user_location
    result = graph.invoke(payload, config=_build_graph_config(thread_id))
    response_text = _extract_response_text(result)
    if not response_text:
        return {"text": "I'm having trouble with that. Please try again.", "hangup": False}
    hangup = bool(result.get("should_end", False))
    log_event("turn_end", thread_id=thread_id, response=response_text[:200], hangup=hangup)
    logger.info(f"{thread_id} response: {response_text[:100]}")
    return {"text": response_text, "hangup": hangup}


# ── Twilio Media Streams WebSocket ────────────────────────────────────────────

@app.websocket("/twilio/stream")
async def twilio_stream(ws: WebSocket):
    """
    Bidirectional Twilio Media Stream.

    Per-turn flow:
      Twilio mulaw 8kHz → Deepgram v2 streaming (your key)
      → EndOfTurn transcript → LangGraph agent → ElevenLabs mulaw 8kHz → caller

    Uses Deepgram v2's conversational turn detection:
    - EagerEndOfTurn  →  start agent early (moderate confidence, lower latency)
    - TurnResumed     →  cancel if user is still speaking
    - EndOfTurn       →  definitive finish
    """
    await ws.accept()
    loop = asyncio.get_running_loop()

    if not DEEPGRAM_API_KEY:
        logger.error("DEEPGRAM_API_KEY not configured — closing stream")
        await ws.close()
        return

    stream_sid: str | None = None
    call_sid: str | None = None
    from_number: str | None = None
    call_started_at: float = time.time()

    # Transcripts from Deepgram → agent worker
    transcript_q: asyncio.Queue[str | None] = asyncio.Queue()

    # Set when Deepgram fires TurnResumed (user kept speaking after EagerEndOfTurn).
    # Shared between async _dg_receiver and threaded sentence_cb, so use threading.Event.
    turn_cancelled = threading.Event()
    # Set while agent audio chunks are actively being sent to Twilio.
    is_playing = threading.Event()

    # ── Agent + TTS worker ────────────────────────────────────────────────────

    async def _agent_worker():
        while True:
            transcript = await transcript_q.get()
            if transcript is None:
                break

            if not _check_and_register(call_sid or "", transcript):
                logger.info(f"Duplicate [{call_sid}], skipping")
                continue

            turn_cancelled.clear()
            timing.reset()
            t_turn_start = time.time()
            turn_number = timing._call["turn_count"] + 1
            logger.info(f"─── TURN {turn_number} ──────────────────────────────────────────")
            log_event("turn_start", thread_id=call_sid or "", transcript=transcript[:200])

            # Single queue drains all audio: filler → sentence-streamed response (or fallback TTS).
            audio_q: asyncio.Queue[bytes | None] = asyncio.Queue()
            # Set by sentence_cb(None) when streaming completes; checked after graph returns.
            sentence_did_stream = threading.Event()

            bytes_sent = 0

            async def _drain_audio() -> None:
                nonlocal bytes_sent
                while True:
                    chunk = await audio_q.get()
                    if chunk is None:
                        is_playing.clear()
                        break
                    if turn_cancelled.is_set():
                        continue  # drain the queue but don't send
                    if stream_sid:
                        try:
                            if not is_playing.is_set():
                                logger.info(f"Agent audio started [{call_sid}]")
                                is_playing.set()
                            await ws.send_json({
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {"payload": base64.b64encode(chunk).decode()},
                            })
                            bytes_sent += len(chunk)
                        except Exception as e:
                            logger.error(f"Audio send failed [{call_sid}]: {e}")
                is_playing.clear()

            drain_task = asyncio.create_task(_drain_audio())

            def early_speak_cb(text: str) -> None:
                """Generate filler TTS synchronously in the graph thread and queue chunks."""
                try:
                    t = time.time()
                    for chunk in _tts_mulaw_iter(text):
                        if chunk:
                            loop.call_soon_threadsafe(audio_q.put_nowait, chunk)
                    timing.add_tts(time.time() - t)
                except Exception as e:
                    logger.error(f"Early speak TTS failed: {e}")

            def sentence_cb(text_or_none: str | None) -> None:
                """Generate per-sentence TTS in the graph thread; None signals end of response."""
                if text_or_none is None:
                    sentence_did_stream.set()
                    loop.call_soon_threadsafe(audio_q.put_nowait, None)
                    return
                if turn_cancelled.is_set():
                    return
                try:
                    t = time.time()
                    for chunk in _tts_mulaw_iter(text_or_none):
                        if chunk:
                            loop.call_soon_threadsafe(audio_q.put_nowait, chunk)
                    timing.add_tts(time.time() - t)
                except Exception as e:
                    logger.error(f"Sentence TTS failed: {e}")

            reply: dict = {"text": "", "hangup": False}
            try:
                with bind_speak_early_callback(early_speak_cb):
                    with bind_sentence_callback(sentence_cb):
                        # loop.run_in_executor doesn't propagate contextvars to the
                        # worker thread — carry the bound callbacks over explicitly.
                        ctx = contextvars.copy_context()
                        reply = await loop.run_in_executor(
                            _executor,
                            lambda: ctx.run(_graph_reply, transcript, call_sid or "", from_number),
                        )
            except GraphRecursionError as e:
                log_event("graph_error", thread_id=call_sid or "", error_type="GraphRecursionError", message=str(e))
                reply = {"text": "I'm getting a bit turned around. Could you re-state what you need?", "hangup": False}
            except Exception as e:
                log_event("graph_error", thread_id=call_sid or "", error_type=type(e).__name__, message=str(e))
                logger.error(f"Agent error [{call_sid}]: {e}", exc_info=True)
                reply = {"text": "Something went wrong. Please try again or ask airport staff for help.", "hangup": False}

            # If TurnResumed fired while we were processing, the user kept speaking.
            # Discard this response and let the subsequent EndOfTurn handle the full utterance.
            if turn_cancelled.is_set():
                logger.info(f"Turn cancelled (TurnResumed) [{call_sid}], discarding response")
                if not sentence_did_stream.is_set():
                    audio_q.put_nowait(None)
                await drain_task
                continue

            # If sentence streaming didn't complete (error path, or graph emitted no text),
            # fall back to full-text TTS so the drain task always receives a None terminator.
            if not sentence_did_stream.is_set():
                if stream_sid and reply.get("text"):
                    try:
                        t_tts = time.time()
                        chunks = await loop.run_in_executor(
                            _executor, lambda: list(_tts_mulaw_iter(reply["text"]))
                        )
                        for chunk in chunks:
                            if chunk:
                                audio_q.put_nowait(chunk)
                        timing.add_tts(time.time() - t_tts)
                    except Exception as e:
                        logger.error(f"Fallback TTS failed [{call_sid}]: {e}")
                audio_q.put_nowait(None)

            await drain_task

            turn_total = time.time() - t_turn_start
            logger.info(timing.summary(turn_total))
            logger.info("─────────────────────────────────────────────────────────")
            log_event(
                "turn_latency",
                thread_id=call_sid or "",
                latency_ms=turn_total * 1000,
                hangup=bool(reply.get("hangup")),
            )
            try:
                await loop.run_in_executor(_executor, lambda: insert_turn(
                    call_id=call_sid or str(uuid.uuid4()),
                    turn_number=turn_number,
                    turn_stats=timing.turn_snapshot(turn_total),
                    tools_used=get_turn_tools_used(),
                ))
            except Exception:
                logger.exception("insert_turn failed in agent_worker")

            if reply.get("hangup") and _twilio and call_sid:
                # Wait for Twilio to finish playing buffered audio before hanging up.
                # At 8 kHz μ-law, 1 byte = 1 sample = 1/8000 s. Add 0.5 s network buffer.
                if bytes_sent > 0:
                    await asyncio.sleep(bytes_sent / 8000.0 + 0.5)
                try:
                    await loop.run_in_executor(
                        _executor,
                        lambda: _twilio.calls(call_sid).update(status="completed"),
                    )
                except Exception as e:
                    logger.error(f"Hangup failed [{call_sid}]: {e}")

    worker = asyncio.create_task(_agent_worker())

    # ── Deepgram v2 streaming connection ─────────────────────────────────────

    dg_client = AsyncDeepgramClient(api_key=DEEPGRAM_API_KEY)

    try:
        async with dg_client.listen.v2.connect(
            model="flux-general-en",
            encoding="mulaw",
            sample_rate=8000,
        ) as dg_socket:

            # Receive Deepgram transcripts in a background task
            async def _dg_receiver():
                try:
                    async for msg in dg_socket:
                        # SDK may return a plain dict instead of ListenV2TurnInfo
                        if isinstance(msg, dict):
                            if msg.get("type") != "TurnInfo":
                                continue
                            event = msg.get("event", "")
                            transcript = (msg.get("transcript") or "").strip()
                        elif isinstance(msg, ListenV2TurnInfo):
                            event = msg.event
                            transcript = msg.transcript.strip()
                        else:
                            continue

                        if event == "TurnResumed":
                            # User continued speaking after EagerEndOfTurn — cancel in-flight response.
                            logger.info(f"Deepgram [TurnResumed]: cancelling in-flight turn [{call_sid}]")
                            turn_cancelled.set()
                            if stream_sid:
                                try:
                                    await ws.send_json({"event": "clear", "streamSid": stream_sid})
                                except Exception as e:
                                    logger.error(f"Twilio clear on TurnResumed failed [{call_sid}]: {e}")
                        elif event in ("EndOfTurn", "EagerEndOfTurn"):
                            logger.info(f"Deepgram [{event}]: '{transcript}'")
                            if transcript:
                                # Barge-in: if TTS is playing when user speech is confirmed, stop it.
                                if is_playing.is_set() and not turn_cancelled.is_set():
                                    logger.info(f"Barge-in on {event} [{call_sid}]")
                                    turn_cancelled.set()
                                    if stream_sid:
                                        try:
                                            await ws.send_json({"event": "clear", "streamSid": stream_sid})
                                        except Exception as e:
                                            logger.error(f"Twilio clear on barge-in failed [{call_sid}]: {e}")
                                await transcript_q.put(transcript)
                except Exception as e:
                    logger.error(f"_dg_receiver error: {e}", exc_info=True)
                logger.info("_dg_receiver done")

            dg_task = asyncio.create_task(_dg_receiver())

            # Forward Twilio audio → Deepgram
            media_count = 0
            try:
                async for raw in ws.iter_text():
                    msg = json.loads(raw)
                    event = msg.get("event")

                    if event == "start":
                        stream_sid = msg["streamSid"]
                        call_sid = msg["start"]["callSid"]
                        custom_params = msg["start"].get("customParameters", {})
                        from_number = custom_params.get("from_number")
                        call_type = custom_params.get("call_type", "inbound")
                        initial_context = custom_params.get("initial_context", "")
                        call_started_at = time.time()
                        timing.call_reset()
                        _uid_hash = hash_user_id(from_number) if from_number else None
                        await loop.run_in_executor(_executor, lambda: start_call(
                            call_id=call_sid,
                            user_id_hash=_uid_hash,
                            started_at=call_started_at,
                        ))
                        logger.info(f"Stream started: {stream_sid} call={call_sid} from={from_number} type={call_type}")
                        log_event("call_start", thread_id=call_sid or "", from_number=from_number or "")

                        # For outbound calls the agent speaks first — inject the notification
                        # context as the initial "user" message so the agent opens the call.
                        if call_type == "outbound" and initial_context:
                            await transcript_q.put(initial_context)

                    elif event == "media":
                        media_count += 1
                        if media_count % 50 == 1:
                            logger.debug(f"Media chunks forwarded to Deepgram: {media_count}")
                        payload = base64.b64decode(msg["media"]["payload"])
                        await dg_socket.send_media(payload)
                        # Barge-in VAD: log ratio when agent is speaking so we can tune the threshold.
                        if is_playing.is_set() and not turn_cancelled.is_set():
                            ratio = _mulaw_voice_ratio(payload)
                            if ratio > 0.02:
                                logger.info(f"VAD ratio={ratio:.2f} len={len(payload)} [{call_sid}]")
                            if ratio > _VAD_THRESHOLD:
                                logger.info(f"Barge-in detected (ratio={ratio:.2f}) [{call_sid}]")
                                turn_cancelled.set()
                                if stream_sid:
                                    try:
                                        await ws.send_json({"event": "clear", "streamSid": stream_sid})
                                    except Exception as e:
                                        logger.error(f"Twilio clear on barge-in failed [{call_sid}]: {e}")

                    elif event == "stop":
                        logger.info(f"Stream stopped: {stream_sid} (total media chunks: {media_count})")
                        break

            except WebSocketDisconnect:
                logger.info(f"Twilio WebSocket disconnected: {stream_sid}")
            except Exception as e:
                logger.error(f"Stream receive error [{stream_sid}]: {e}", exc_info=True)
            finally:
                dg_task.cancel()
                await dg_socket.send_close_stream()

    except Exception as e:
        logger.error(f"Deepgram connection failed: {e}", exc_info=True)
    finally:
        await transcript_q.put(None)  # stop agent worker
        await worker
        log_event("call_end", thread_id=call_sid or "", from_number=from_number or "")
        logger.info(f"Stream cleaned up: {stream_sid}")

        # Write analytics — non-blocking, never raises
        try:
            duration = time.time() - call_started_at
            cfg = {"configurable": {"thread_id": call_sid}} if call_sid else None
            state_snap: dict = {}
            if cfg:
                snap = graph.get_state(cfg)
                state_snap = snap.values if snap else {}
            messages = state_snap.get("messages", [])
            call_stats = timing.call_snapshot()
            summary_data = await loop.run_in_executor(
                _executor, summarize, messages, state_snap
            )
            await loop.run_in_executor(
                _executor, lambda: finish_call(
                    call_id=call_sid or str(uuid.uuid4()),
                    duration_s=duration,
                    turn_count=call_stats["turn_count"],
                    timing=call_stats,
                    flight_number=state_snap.get("flight_number"),
                    topics=summary_data["topics"],
                    resolved=summary_data["resolved"],
                    summary=summary_data["summary"],
                )
            )
        except Exception:
            logger.exception(f"Analytics write failed for call {call_sid}")


# ── Web Push-to-Talk WebSocket ────────────────────────────────────────────────

@app.websocket("/web/stream")
async def web_stream(ws: WebSocket):
    """
    Browser always-on voice endpoint (no Twilio).

    Protocol:
      Client → Server (JSON):
        { "type": "config",      "language": "en", "userId": "..." }
        { "type": "audio",       "data": "<base64 webm/opus>", "language": "en", "format": "webm" }
        { "type": "audio_start", "language": "en" }               — begin a live-transcribed utterance
        { "type": "audio_chunk", "data": "<base64 webm/opus>" }   — repeated small chunks while speaking
        { "type": "audio_end" }                                   — VAD detected silence; finalize + run turn
        { "type": "text",        "text": "...", "language": "en" }

      Server → Client (JSON):
        { "type": "transcript",         "role": "user"|"agent", "text": "..." }
        { "type": "partial_transcript", "text": "...", "final": false }  — live growing transcript while user speaks
        { "type": "audio",              "data": "<base64 mp3>" }
        { "type": "status",             "state": "idle"|"thinking"|"speaking", "label": "..." }
        { "type": "map_action",         "action": {...} }  — show/clear a destination, route, or directions on the map
        { "type": "error",              "message": "..." }
    """
    if not _origin_allowed(ws.headers.get("origin")):
        await ws.close(code=1008)
        return

    await ws.accept()

    loop = asyncio.get_running_loop()
    session_id = str(uuid.uuid4())
    language = "en"
    user_id: str | None = None
    latest_location: dict | None = None  # {lat, lng, accuracy, ts} — most recent GPS fix from the client

    # Per-utterance Deepgram live-streaming state (see audio_start/audio_chunk/audio_end below)
    dg_client = AsyncDeepgramClient(api_key=DEEPGRAM_API_KEY) if DEEPGRAM_API_KEY else None
    dg_cm = None
    dg_socket = None
    dg_recv_task: asyncio.Task | None = None
    dg_final_holder: dict | None = None
    dg_final_event: asyncio.Event | None = None

    logger.info(f"Web stream connected: {session_id}")

    async def _send(msg: dict) -> None:
        try:
            await ws.send_json(msg)
        except Exception:
            pass

    async def _close_dg_socket() -> None:
        """Tear down any in-flight Deepgram live connection (best-effort)."""
        nonlocal dg_cm, dg_socket, dg_recv_task
        if dg_recv_task is not None:
            dg_recv_task.cancel()
            dg_recv_task = None
        if dg_cm is not None:
            try:
                await dg_cm.__aexit__(None, None, None)
            except Exception:
                pass
            dg_cm = None
        dg_socket = None

    async def _transcribe_audio(audio_bytes: bytes, fmt: str) -> str | None:
        """Send audio bytes to Deepgram REST API for transcription."""
        if not DEEPGRAM_API_KEY:
            logger.error("DEEPGRAM_API_KEY not set — cannot transcribe")
            return None
        try:
            import httpx
            mime_map = {"webm": "audio/webm", "ogg": "audio/ogg", "mp4": "audio/mp4"}
            mime = mime_map.get(fmt, "audio/webm")

            lang_code = language if language not in ("auto", "") else None
            params: dict = {"model": "nova-3", "smart_format": "true", "punctuate": "true"}
            if lang_code:
                params["language"] = lang_code
            else:
                params["detect_language"] = "true"

            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(
                    "https://api.deepgram.com/v1/listen",
                    headers={
                        "Authorization": f"Token {DEEPGRAM_API_KEY}",
                        "Content-Type": mime,
                    },
                    content=audio_bytes,
                    params=params,
                )
                resp.raise_for_status()
                data = resp.json()

            channels = data.get("results", {}).get("channels", [{}])
            alts = channels[0].get("alternatives", [{}])
            transcript = alts[0].get("transcript", "").strip()
            logger.info(f"Deepgram transcript: '{transcript}'")
            return transcript or None
        except Exception as e:
            logger.error(f"Deepgram transcription failed: {e}")
            return None

    async def _tts_mp3_bytes(text: str) -> bytes | None:
        """Generate MP3 bytes using configured TTS provider."""
        try:
            if TTS_PROVIDER == "elevenlabs" and _eleven and ELEVENLABS_VOICE_ID:
                audio_bytes = b"".join(
                    await loop.run_in_executor(
                        _executor,
                        lambda: list(_eleven.text_to_speech.convert(
                            voice_id=ELEVENLABS_VOICE_ID,
                            text=text,
                            model_id="eleven_flash_v2_5",
                            output_format="mp3_44100_128",
                        ))
                    )
                )
                return audio_bytes
            elif CARTESIA_API_KEY:
                resp = await loop.run_in_executor(
                    _executor,
                    lambda: _requests.post(
                        _CARTESIA_URL,
                        headers={**_CARTESIA_HEADERS, "X-API-Key": CARTESIA_API_KEY},
                        json={
                            "model_id": CARTESIA_MODEL_ID,
                            "transcript": text,
                            "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
                            "output_format": {"container": "mp3", "encoding": "mp3", "sample_rate": 44100},
                            "language": language if len(language) == 2 else "en",
                            "generation_config": {"emotion": CARTESIA_EMOTION, "speed": "fast"},
                        },
                        timeout=15,
                    )
                )
                resp.raise_for_status()
                return resp.content
        except Exception as e:
            logger.error(f"TTS failed: {e}")
        return None

    async def _run_turn(user_text: str) -> None:
        """Run the agent graph on already-transcribed text and speak the reply."""
        await _send({"type": "status", "state": "thinking"})

        def tool_status_cb(tool_name: str) -> None:
            """Fires from the graph's executor thread the moment a tool call
            is about to run — relay a short status label to the browser."""
            label = _TOOL_STATUS_LABELS.get(tool_name)
            if not label:
                return
            asyncio.run_coroutine_threadsafe(
                _send({"type": "status", "state": "thinking", "label": label}),
                loop,
            )

        def map_cb(action: dict) -> None:
            """Fires from the graph's executor thread whenever a map tool
            (show_map_destination, show_map_directions, ...) runs — relay the
            action to the browser immediately so the map updates live.
            checkpoint_prompt/checkpoint_resolved aren't map redraws — they're
            conversational UI state — so they go out as their own top-level
            message type instead of being wrapped in map_action."""
            if action.get("type") in ("checkpoint_prompt", "checkpoint_resolved"):
                payload = {k: v for k, v in action.items()}
            else:
                payload = {"type": "map_action", "action": action}
            asyncio.run_coroutine_threadsafe(_send(payload), loop)

        try:
            with bind_tool_status_callback(tool_status_cb):
                with bind_map_callback(map_cb):
                    # loop.run_in_executor doesn't propagate contextvars to the
                    # worker thread, so the callbacks bound just above wouldn't
                    # be visible inside _graph_reply unless we carry the context
                    # over explicitly via copy_context().run(...).
                    ctx = contextvars.copy_context()
                    reply = await loop.run_in_executor(
                        _executor,
                        lambda t=user_text, loc=latest_location: ctx.run(_graph_reply, t, session_id, user_id, loc)
                    )
            agent_text = reply["text"]
        except GraphRecursionError:
            agent_text = "I'm having trouble with that right now. Can you rephrase?"
        except Exception as e:
            logger.error(f"Graph error: {e}", exc_info=True)
            agent_text = "Something went wrong. Please try again."

        await _send({"type": "transcript", "role": "agent", "text": agent_text})

        await _send({"type": "status", "state": "speaking"})
        mp3_bytes = await _tts_mp3_bytes(agent_text)
        if mp3_bytes:
            await _send({"type": "audio", "data": base64.b64encode(mp3_bytes).decode()})

        await _send({"type": "status", "state": "idle"})

    async def _dg_receiver(sock, holder: dict, done: asyncio.Event) -> None:
        """Relay Deepgram live results back to the client as partial_transcript
        messages, accumulating finalized fragments into `holder["text"]`."""
        try:
            async for m in sock:
                if getattr(m, "type", None) != "Results":
                    continue
                alt = m.channel.alternatives[0]
                text = alt.transcript
                if not text:
                    if m.speech_final:
                        done.set()
                    continue

                if m.is_final:
                    holder["text"] = f"{holder['text']} {text}".strip()
                    await _send({"type": "partial_transcript", "text": holder["text"], "final": bool(m.speech_final)})
                    if m.speech_final:
                        done.set()
                else:
                    preview = f"{holder['text']} {text}".strip()
                    await _send({"type": "partial_transcript", "text": preview, "final": False})
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Deepgram live receiver error [{session_id}]: {e}")

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "config":
                language = msg.get("language", "en")
                user_id = msg.get("userId") or user_id
                logger.info(f"Web stream [{session_id}] config: lang={language} user={user_id}")

                # Send greeting on first config message
                greeting = "Welcome to Oakland International Airport! I'm your AI guide. Ask me anything — gates, flights, restaurants, restrooms, or directions anywhere in the terminal."
                await _send({"type": "transcript", "role": "agent", "text": greeting})
                await _send({"type": "status", "state": "speaking"})
                mp3_bytes = await _tts_mp3_bytes(greeting)
                if mp3_bytes:
                    await _send({"type": "audio", "data": base64.b64encode(mp3_bytes).decode()})
                await _send({"type": "status", "state": "idle"})

            elif msg_type == "audio":
                audio_b64 = msg.get("data", "")
                fmt = msg.get("format", "webm")
                language = msg.get("language", language)

                if not audio_b64:
                    await _send({"type": "error", "message": "Empty audio data"})
                    continue

                audio_bytes = base64.b64decode(audio_b64)

                await _send({"type": "status", "state": "thinking"})
                transcript = await _transcribe_audio(audio_bytes, fmt)

                if not transcript:
                    await _send({"type": "error", "message": "Could not understand audio. Please try again."})
                    await _send({"type": "status", "state": "idle"})
                    continue

                logger.info(f"Web [{session_id}] transcript: {transcript[:100]}")
                await _send({"type": "transcript", "role": "user", "text": transcript})
                await _run_turn(transcript)

            elif msg_type == "audio_start":
                language = msg.get("language", language)

                if dg_client is None:
                    await _send({"type": "error", "message": "Speech recognition is not configured."})
                    continue

                await _close_dg_socket()  # safety net if a previous utterance wasn't cleanly closed

                dg_cm = dg_client.listen.v1.connect(
                    model="nova-3",
                    interim_results=True,
                    punctuate=True,
                    smart_format=True,
                    language=language if language not in ("auto", "") else None,
                )
                try:
                    dg_socket = await dg_cm.__aenter__()
                except Exception as e:
                    logger.error(f"Deepgram live connect failed [{session_id}]: {e}")
                    dg_cm = None
                    dg_socket = None
                    # Without this, the client's VAD still runs its full
                    # start/silence cycle and sends audio_end expecting a
                    # reply that never comes — it hangs in "thinking" with
                    # no transcript and no error, forever.
                    await _send({"type": "error", "message": "Couldn't connect to speech recognition. Please try again."})
                    await _send({"type": "status", "state": "idle"})
                    continue

                dg_final_holder = {"text": ""}
                dg_final_event = asyncio.Event()
                dg_recv_task = asyncio.create_task(_dg_receiver(dg_socket, dg_final_holder, dg_final_event))

            elif msg_type == "audio_chunk":
                chunk_b64 = msg.get("data", "")
                if dg_socket is not None and chunk_b64:
                    try:
                        await dg_socket.send_media(base64.b64decode(chunk_b64))
                    except Exception as e:
                        # The live socket died mid-utterance (e.g. a keepalive
                        # ping timeout) — tear it down now instead of retrying
                        # every subsequent chunk against a dead connection,
                        # which just floods the log and silently loses audio
                        # until audio_end's 3s finalize timeout.
                        logger.error(f"Deepgram send_media failed [{session_id}]: {e}")
                        await _close_dg_socket()
                        await _send({"type": "error", "message": "Speech recognition connection dropped. Please try again."})
                        await _send({"type": "status", "state": "idle"})

            elif msg_type == "audio_end":
                if dg_socket is None:
                    # No live socket (connect failed earlier, or was never
                    # opened) — tell the client rather than leaving it
                    # hanging in "thinking" with nothing ever arriving.
                    await _send({"type": "error", "message": "Speech recognition wasn't active for that. Please try again."})
                    await _send({"type": "status", "state": "idle"})
                    continue

                try:
                    await dg_socket.send_finalize()
                    if dg_final_event is not None:
                        try:
                            await asyncio.wait_for(dg_final_event.wait(), timeout=3.0)
                        except asyncio.TimeoutError:
                            pass
                except Exception as e:
                    logger.error(f"Deepgram finalize failed [{session_id}]: {e}")

                transcript = (dg_final_holder or {}).get("text", "").strip()
                await _close_dg_socket()
                dg_final_holder = None
                dg_final_event = None

                if not transcript:
                    await _send({"type": "error", "message": "Could not understand audio. Please try again."})
                    await _send({"type": "status", "state": "idle"})
                    continue

                logger.info(f"Web [{session_id}] live transcript: {transcript[:100]}")
                await _send({"type": "transcript", "role": "user", "text": transcript})
                await _run_turn(transcript)

            elif msg_type == "text":
                text_in = (msg.get("text") or "").strip()
                language = msg.get("language", language)

                if not text_in:
                    await _send({"type": "error", "message": "Empty text"})
                    continue

                await _send({"type": "transcript", "role": "user", "text": text_in})
                await _run_turn(text_in)

            elif msg_type == "location":
                lat, lng = msg.get("lat"), msg.get("lng")
                if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
                    latest_location = {
                        "lat": lat,
                        "lng": lng,
                        "accuracy": msg.get("accuracy"),
                        "ts": time.time(),
                    }

            elif msg_type == "checkpoint_ack":
                # Button-tap confirmation — turned into a synthetic user turn so
                # the exact same LLM judgment (incl. the GPS sanity-check in
                # advance_checkpoint's own instructions) governs both the
                # voice and button confirmation paths; no special-cased logic here.
                poi_name = msg.get("poiName") or "the checkpoint"
                synthetic_text = f"[checkpoint confirmed via button: user reached {poi_name}]"
                await _run_turn(synthetic_text)

            elif msg_type == "checkpoint_help":
                # "Need help" tap — doesn't advance anything, just tells the
                # agent (as a synthetic turn) the user is stuck at the current
                # checkpoint so it can give more detail without losing their place.
                poi_name = msg.get("poiName") or "the current checkpoint"
                synthetic_text = f"[user tapped 'need help' — they can't find {poi_name}]"
                await _run_turn(synthetic_text)

    except WebSocketDisconnect:
        logger.info(f"Web stream disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Web stream error [{session_id}]: {e}", exc_info=True)
        try:
            await _send({"type": "error", "message": "Server error. Please reconnect."})
        except Exception:
            pass
    finally:
        await _close_dg_socket()


# ── Voice entry point ─────────────────────────────────────────────────────────

@app.post("/twilio/voice")
async def twilio_voice_incoming(request: Request):
    """
    Called by Twilio when an inbound call arrives.
    Plays ElevenLabs welcome message, then opens the Media Stream.
    """
    raw_body = await request.body()
    params = _parse_form(raw_body)
    if not _verify_twilio_signature(request, params):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    call_sid = params.get("CallSid", "unknown")
    from_number = params.get("From", "unknown")
    logger.info(f"Incoming call: {call_sid} from {from_number}")

    # Fallback mode: use Twilio's built-in speech recognition if Deepgram is not configured.
    if not DEEPGRAM_API_KEY:
        welcome = "Welcome to Oakland International Airport. How can I help you today?"
        vr = _build_gather_twiml(welcome)
        return Response(content=str(vr), media_type="application/xml")

    welcome = "Welcome to Oakland International Airport. How can I help you today?"
    audio_url = await asyncio.get_running_loop().run_in_executor(_executor, _tts_mp3_url, welcome)

    vr = VoiceResponse()
    if audio_url:
        vr.play(audio_url)
    else:
        vr.say(welcome, voice=_FALLBACK_VOICE)

    ws_url = SERVER_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
    connect = Connect()
    stream = Stream(url=f"{ws_url}/twilio/stream")
    stream.parameter(name="from_number", value=from_number)
    connect.append(stream)
    vr.append(connect)

    return Response(content=str(vr), media_type="application/xml")


@app.post("/twilio/voice/process")
async def twilio_voice_process(request: Request):
    """
    Twilio Gather callback for built-in speech recognition fallback mode.
    """
    raw_body = await request.body()
    params = _parse_form(raw_body)
    if not _verify_twilio_signature(request, params):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    call_sid = params.get("CallSid", "unknown")
    from_number = params.get("From", "unknown")
    transcript = (params.get("SpeechResult") or "").strip()

    if not transcript:
        vr = _build_gather_twiml("I didn't catch that. Please say that again.")
        return Response(content=str(vr), media_type="application/xml")

    loop = asyncio.get_running_loop()
    timing.reset()
    try:
        reply = await loop.run_in_executor(
            _executor, _graph_reply, transcript, call_sid, from_number
        )
        reply_text = reply.get("text") or "I'm sorry, I had trouble with that."
    except Exception as e:
        logger.error(f"Twilio gather graph failed [{call_sid}]: {e}", exc_info=True)
        reply_text = "Something went wrong. Please try again."
        reply = {"hangup": False}

    vr = VoiceResponse()
    vr.say(reply_text, voice=_FALLBACK_VOICE)
    if reply.get("hangup"):
        vr.hangup()
    else:
        vr.redirect("/twilio/voice")
    return Response(content=str(vr), media_type="application/xml")


# ── SMS endpoint ──────────────────────────────────────────────────────────────

@app.post("/twilio/sms")
async def twilio_sms(request: Request):
    """Handle incoming SMS messages."""
    raw_body = await request.body()
    params = _parse_form(raw_body)
    if not _verify_twilio_signature(request, params):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    from_number = params.get("From", "")
    body_text = params.get("Body", "").strip()
    logger.info(f"SMS from {from_number}: '{body_text}'")

    if not body_text:
        mr = MessagingResponse()
        mr.message("I didn't receive your message. Please try again.")
        return Response(content=str(mr), media_type="application/xml")

    loop = asyncio.get_running_loop()

    # Start a new SMS session if needed
    if from_number not in _sms_sessions:
        _sms_sessions[from_number] = {"started_at": time.time(), "turn_count": 0}
        timing.call_reset()

    # END (all caps) terminates the session and writes analytics
    if body_text == "END":
        session = _sms_sessions.pop(from_number, {})
        started_at = session.get("started_at", time.time())
        try:
            cfg = {"configurable": {"thread_id": from_number}}
            snap = graph.get_state(cfg)
            state_snap: dict = snap.values if snap else {}
            messages = state_snap.get("messages", [])
            call_stats = timing.call_snapshot()
            summary_data = await loop.run_in_executor(_executor, summarize, messages, state_snap)
            uid_hash = hash_user_id(from_number) if from_number else None
            await loop.run_in_executor(
                _executor, lambda: insert_call(
                    call_id=str(uuid.uuid4()),
                    airport_id=DEFAULT_AIRPORT,
                    user_id_hash=uid_hash,
                    started_at=started_at,
                    duration_s=time.time() - started_at,
                    turn_count=call_stats["turn_count"],
                    timing=call_stats,
                    flight_number=state_snap.get("flight_number"),
                    topics=summary_data["topics"],
                    resolved=summary_data["resolved"],
                    summary=summary_data["summary"],
                )
            )
        except Exception:
            logger.exception(f"SMS analytics write failed for {from_number}")
        mr = MessagingResponse()
        mr.message("Thanks, goodbye!")
        return Response(content=str(mr), media_type="application/xml")

    _sms_sessions[from_number]["turn_count"] += 1
    timing.reset()
    try:
        reply = await loop.run_in_executor(_executor, _graph_reply, body_text, from_number, from_number)
    except Exception as e:
        logger.error(f"SMS graph failed [{from_number}]: {e}", exc_info=True)
        reply = {"text": "Something went wrong. Please try again."}

    mr = MessagingResponse()
    mr.message(reply["text"])
    return Response(content=str(mr), media_type="application/xml")


# ── SMS invite (web UI "Text me instead") ────────────────────────────────────

class SmsInviteRequest(BaseModel):
    to: str  # E.164 phone number e.g. "+14155551234"

@app.post("/sms-invite")
async def sms_invite(req: SmsInviteRequest):
    """Send a welcome SMS so the user can continue the conversation via text."""
    if not _twilio or not TWILIO_PHONE_NUMBER:
        raise HTTPException(status_code=503, detail="SMS not configured")
    try:
        body = (
            f"👋 Hi! You can chat with the Oakland Airport assistant right here via text. "
            f"Just send us a message and we'll help you navigate the airport, find your gate, check flight status, and more!"
        )
        _twilio.messages.create(body=body, from_=TWILIO_PHONE_NUMBER, to=req.to)
        return {"ok": True}
    except Exception as e:
        logger.error(f"SMS invite failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Gate change notifications (outbound) ──────────────────────────────────────

class GateChangeRequest(BaseModel):
    phone: str
    flight: str
    old_gate: str
    new_gate: str


@app.post("/gate-change")
async def gate_change(body: GateChangeRequest):
    """Notify a passenger of a gate change via outbound call or SMS fallback."""
    message = (
        f"Hi, this is Oakland International Airport. Your flight {body.flight} "
        f"gate has changed from {body.old_gate} to {body.new_gate}. "
        f"Please proceed to gate {body.new_gate}. Thank you."
    )

    if not _twilio or not TWILIO_PHONE_NUMBER:
        logger.info(f"[MOCK] → {body.phone}: {message}")
        return {"status": "mock_notified", "message": message}

    loop = asyncio.get_running_loop()
    audio_url = await loop.run_in_executor(_executor, _tts_mp3_url, message)
    twiml = (
        f"<Response><Play>{audio_url}</Play><Hangup/></Response>"
        if audio_url
        else f'<Response><Say voice="{_FALLBACK_VOICE}">{message}</Say><Hangup/></Response>'
    )

    try:
        call = await loop.run_in_executor(
            _executor,
            lambda: _twilio.calls.create(to=body.phone, from_=TWILIO_PHONE_NUMBER, twiml=twiml),
        )
        return {"status": "called", "call_sid": call.sid}
    except Exception as e:
        logger.error(f"Outbound call failed: {e}")

    try:
        sms = await loop.run_in_executor(
            _executor,
            lambda: _twilio.messages.create(to=body.phone, from_=TWILIO_PHONE_NUMBER, body=message),
        )
        return {"status": "sms_sent", "message_sid": sms.sid}
    except Exception as e:
        logger.error(f"SMS fallback failed: {e}")
        return {"status": "error", "error": str(e)}


# ── Flight-change outbound agent call ─────────────────────────────────────────

class FlightChangeCallRequest(BaseModel):
    phone: str
    flight: str
    airport: str = DEFAULT_AIRPORT
    changes: list[dict]  # [{field, old_value, new_value}, ...]


def _build_outbound_context(flight: str, changes: list[dict]) -> str:
    """Build a concise context string injected as the agent's first message."""
    parts = []
    for c in changes:
        field = c.get("field", "status")
        old_v = c.get("old_value") or "unknown"
        new_v = c.get("new_value") or "unknown"
        parts.append(f"{field} changed from {old_v} to {new_v}")
    change_desc = "; ".join(parts) if parts else "status updated"
    return (
        f"[OUTBOUND FLIGHT NOTIFICATION] You are calling a passenger on behalf of Oakland "
        f"International Airport. Flight {flight}: {change_desc}. "
        f"Greet them, inform them about this update concisely, and offer to help navigate "
        f"to the new gate or answer any questions they have."
    )


@app.post("/flight-change-call")
async def flight_change_call(body: FlightChangeCallRequest):
    """Make an outbound call with the full voice agent to notify a passenger of flight changes.

    Called by gategetter's auto_notify when significant changes are detected for a tracked flight.
    The agent speaks first, delivering the notification and offering navigation help.
    """
    context = _build_outbound_context(body.flight, body.changes)

    if not _twilio or not TWILIO_PHONE_NUMBER:
        logger.info(f"[MOCK] Outbound agent call to {body.phone}: {context}")
        return {"status": "mock", "context": context}

    loop = asyncio.get_running_loop()
    ws_url = SERVER_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")

    vr = VoiceResponse()
    connect = Connect()
    stream = Stream(url=f"{ws_url}/twilio/stream")
    stream.parameter(name="from_number", value=body.phone)
    stream.parameter(name="call_type", value="outbound")
    stream.parameter(name="initial_context", value=context)
    connect.append(stream)
    vr.append(connect)
    twiml_str = str(vr)

    try:
        call = await loop.run_in_executor(
            _executor,
            lambda: _twilio.calls.create(to=body.phone, from_=TWILIO_PHONE_NUMBER, twiml=twiml_str),
        )
        logger.info(f"Outbound agent call to {body.phone} for flight {body.flight}: {call.sid}")
        return {"status": "calling", "call_sid": call.sid}
    except Exception as e:
        logger.error(f"Outbound agent call failed for {body.phone}: {e}")
        return {"status": "error", "error": str(e)}


# ── Map cache ─────────────────────────────────────────────────────────────────

class MapRefreshRequest(BaseModel):
    airport_id: str


@app.post("/map/refresh")
async def map_refresh(body: MapRefreshRequest):
    """Force-invalidate the in-process map cache and reload from Supabase."""
    invalidate_map_cache(body.airport_id)
    loop = asyncio.get_running_loop()
    try:
        levels = await loop.run_in_executor(_executor, load_map_levels, body.airport_id)
        return {"status": "refreshed", "airport_id": body.airport_id, "levels": len(levels)}
    except Exception as e:
        logger.error(f"Map refresh failed [{body.airport_id}]: {e}")
        raise HTTPException(status_code=502, detail=str(e))


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "Oakland Airport Agent"}


# ── Flight watch background loop ──────────────────────────────────────────────

# airport → { flight_number → last_scraped_flight_dict }
_flight_watch_prev: dict[str, dict[str, dict]] = {}
_flight_watch_prev_lock = threading.Lock()


async def _flight_watch_loop():
    """Poll gategetter for tracked flights and make outbound agent calls on changes."""
    from agent.flight_tracker import (
        find_significant_changes, get_subscribers, get_tracked, unsubscribe_flight,
    )

    interval = int(os.environ.get("POLL_INTERVAL_SECONDS", "120"))
    logger.info(f"[FlightWatch] Started — polling every {interval}s")
    loop = asyncio.get_running_loop()

    await asyncio.sleep(interval)

    while True:
        try:
            tracked = get_tracked()  # {airport: [flight, ...]}
            for airport, flights in tracked.items():
                if not flights:
                    continue
                try:
                    resp = await loop.run_in_executor(
                        _executor,
                        lambda a=airport: _requests.get(
                            f"{GATEGETTER_URL}/api/data?airport={a}", timeout=45
                        ),
                    )
                    resp.raise_for_status()
                    by_fn: dict[str, dict] = {
                        f["flight_number"].strip().upper().replace(" ", ""): f
                        for f in resp.json().get("flights", [])
                        if f.get("flight_number")
                    }
                except Exception as e:
                    logger.warning(f"[FlightWatch] Could not fetch {airport} data: {e}")
                    continue

                for flight in flights:
                    curr = by_fn.get(flight)
                    with _flight_watch_prev_lock:
                        prev = _flight_watch_prev.get(airport, {}).get(flight)

                    if curr:
                        with _flight_watch_prev_lock:
                            _flight_watch_prev.setdefault(airport, {})[flight] = curr

                        if "depart" in (curr.get("status") or "").lower():
                            unsubscribe_flight(airport, flight)
                            logger.info(f"[FlightWatch] {flight} departed — unsubscribed")
                            continue

                    if prev and curr:
                        changes = find_significant_changes(prev, curr)
                        if not changes:
                            continue
                        phones = get_subscribers(airport, flight)
                        logger.info(f"[FlightWatch] {flight} changed: {changes} → calling {phones}")
                        for phone in phones:
                            try:
                                await flight_change_call(FlightChangeCallRequest(
                                    phone=phone,
                                    flight=flight,
                                    airport=airport,
                                    changes=changes,
                                ))
                            except Exception as e:
                                logger.error(f"[FlightWatch] Outbound call failed {phone}: {e}")

        except Exception as e:
            logger.error(f"[FlightWatch] Loop error: {e}", exc_info=True)

        await asyncio.sleep(interval)


@app.on_event("startup")
async def _startup():
    if os.environ.get("FLIGHT_TRACKER_ENABLED", "false").lower() == "true":
        asyncio.create_task(_flight_watch_loop())
    else:
        logger.info("[FlightWatch] Disabled — set FLIGHT_TRACKER_ENABLED=true to enable")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
