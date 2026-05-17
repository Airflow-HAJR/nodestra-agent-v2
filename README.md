# Lincoln Airport Agent

An AI airport navigation assistant that helps travelers find routes between gates, lounges, and other POIs. Built with LangGraph and Azure OpenAI.

## Features

- **Interactive Navigation** — Ask for directions between any two locations
- **Phone Integration** — Receive incoming calls via AgentPhone webhook
- **Proactive Notifications** — Send gate change alerts via call or SMS
- **Multi-turn Conversations** — Agent remembers your location across turns
- **Error Recovery** — Built-in repair logic for tool failures

## Setup

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required environment variables:
- `AZURE_OPENAI_API_KEY` — Your Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT` — Your Azure OpenAI endpoint
- `AZURE_OPENAI_DEPLOYMENT` — Deployment name (e.g., `gpt-5-mini`)
- `AGENTPHONE_API_KEY` — (Optional) AgentPhone API key for outbound calls/SMS

## Usage

### CLI Mode (Interactive Terminal)

Start an interactive conversation with the agent:

```bash
uv run python main.py
```

Example conversation:
```
User: I'm at gate B4 and need to find the bathroom
Agent: I'll help you find the nearest restroom...
```

### Server Mode (Phone Integration)

Start the FastAPI server to receive calls and send notifications:

```bash
uv run uvicorn server:app --reload --port 8000
```

The server provides three endpoints:

#### 1. **POST `/webhook`** — Incoming Call/SMS Handler

Receives transcribed messages from AgentPhone and routes them through the navigation agent.

**Payload** (from AgentPhone):
```json
{
  "event": "agent.message",
  "channel": "voice",
  "data": {
    "callId": "call_abc123",
    "from": "+15551234567",
    "transcript": "I need directions to gate C7",
    "confidence": 0.95
  }
}
```

**Response**:
Voice replies stream as NDJSON so AgentPhone can start TTS on the first chunk:
```json
{"text": "Let me check that for you.", "interim": true}
{"text": "You're currently at gate B4. To reach gate C7, head south down the main concourse..."}
```

#### 2. **POST `/gate-change`** — Gate Change Notification

Notify a user about a gate change via call or SMS.

**Payload**:
```json
{
  "phone": "+15551234567",
  "flight": "UA123",
  "old_gate": "B4",
  "new_gate": "C7"
}
```

**Response**:
```json
{
  "status": "called",
  "result": { "callId": "call_def456" }
}
```

If the API key is not configured, notifications are logged as mock:
```json
{
  "status": "mock_notified",
  "message": "Hi, this is Oakland Airport. Your flight UA123 gate has changed...",
  "note": "AGENTPHONE_API_KEY not set, using mock notification"
}
```

#### 3. **GET `/health`** — Health Check

```bash
curl http://localhost:8000/health
# Response: { "status": "ok", "service": "Lincoln Airport Agent" }
```

## Testing

Run the included test script to verify endpoints:

```bash
# In one terminal, start the server:
uv run uvicorn server:app --reload

# In another terminal, run the tests:
uv run python test_api.py
```

This will test:
- Health check
- Webhook with a mock voice transcript
- Gate change notification

## Architecture

```
┌──────────────────────────────────────────────────┐
│         LangGraph State Machine                   │
├──────────────────────────────────────────────────┤
│  • Clarify Subgraph  — Determine destination     │
│  • Navigate Subgraph — Provide step-by-step      │
│  • Closure Node      — Detect task completion    │
│  • Error Repair      — Handle tool failures      │
└──────────────────────────────────────────────────┘
         ▲                              │
         │                              │ graphs.invoke()
         │                              ▼
    CLI REPL                    FastAPI Server
    (main.py)                   (server.py)
         │                              │
         └──────────┬───────────────────┘
                    │
          ┌─────────┴─────────┐
          │                   │
      Airport Backend    AgentPhone API
      (REST API)        (Calls/SMS)
```

## Project Structure

```
lincoln/
├── agent/
│   ├── __init__.py
│   ├── config.py          # Environment variables
│   ├── graph.py           # LangGraph state machine
│   ├── llm.py             # LLM factory (Azure OpenAI)
│   ├── prompts.py         # System prompt builder
│   ├── runner.py          # CLI REPL loop
│   ├── state.py           # State schema
│   └── tools.py           # Navigation tools
├── main.py                # CLI entry point
├── server.py              # FastAPI phone server
├── test_api.py            # Integration tests
├── pyproject.toml         # Dependencies
├── .env.example           # Environment template
└── README.md              # This file
```

## How It Works

### Incoming Call Flow

1. User calls the AgentPhone number
2. AgentPhone transcribes speech → POST `/webhook`
3. `callId` is used as `thread_id` for conversation continuity
4. Message goes through LangGraph graph:
   - **Clarify** — Ask where user is and where they want to go
   - **Navigate** — Provide directions and nearby options
   - **Closure** — Detect if conversation is complete
5. Voice turns stream an interim NDJSON chunk first, then the final TTS response

### Outbound Notification Flow

1. External system calls POST `/gate-change` with flight/phone/gate info
2. Message is crafted: "Your flight UA123 gate has changed from B4 to C7..."
3. AgentPhone API is called to make outbound call or SMS
4. Result is logged and returned to caller

### Multi-turn Conversations

Each call (`callId`) becomes a unique `thread_id` in LangGraph. The `InMemorySaver` checkpointer maintains state per thread, so:
- Turn 1: User says "Where's the bathroom?" → Agent asks "Where are you?"
- Turn 2: User says "Gate B4" → State remembers location → Agent provides directions
- Turn 3: User says "What about coffee?" → Agent remembers location → Provides coffee locations

## Tools Available to the Agent

The agent can call these tools to help users:

1. **find_poi** — Fuzzy search for a location (e.g., "bathroom", "Starbucks")
2. **get_route** — Get shortest path between two POI IDs
3. **get_nodes** — List all locations, optionally filtered by floor
4. **resolve_poi** — Disambiguate when multiple locations have the same name
5. **find_nearest** — Find the closest POI of a certain type from user's location
6. **set_nav_state** — Update user's current location

## Future Enhancements

- [ ] Persistent storage (replace `InMemorySaver` with database checkpointer)
- [ ] Multi-language support
- [ ] Real-time flight status integration
- [ ] Accessibility features (braille, enlarged text)
- [ ] Integration with airport wayfinding kiosks
