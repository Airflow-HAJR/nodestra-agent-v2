# Stuttgart Agent Improvements

## 1. Closure Detection In-Prompt (Remove LLM Call)

**What Changed:**
- Removed the separate `_closure_node` that called a second LLM model for classification
- Integrated closure detection directly into the agent system prompt
- Added simple heuristic checking for explicit user decline phrases

**Why:**
- Eliminates one full LLM call per conversation turn (latency + cost)
- Closure detection is straightforward enough to fit in the system prompt
- Agent can now handle conversation endings naturally without context switching

**How It Works:**
- System prompt now explicitly instructs the agent on closure signals and how to respond
- Graph checks for explicit decline phrases (`"no thanks"`, `"bye"`, etc.) as a quick heuristic
- No additional LLM inference needed

**Performance Impact:** ~500-1000ms latency saved per turn (one fewer LLM call)

---

## 2. Consolidated Context Management

**What Changed:**
- Simplified message consolidation in the graph
- Removed redundant closure node and integrated its logic into main agent flow
- Cleaner state transitions: `clarify → navigate → end`

**Why:**
- Fewer nodes = fewer context switches
- Messages are naturally accumulated through `add_messages` in State
- Reduces memory overhead from unnecessary state copies

**Graph Flow:**
```
START → clarify OR navigate → check_if_done → END or continue
```

---

## 3. Connection Pooling for External APIs

**What Changed:**
- Added persistent HTTP session with connection pooling
- Configured pool with 10 base connections and 20 max pool size
- Automatic retry strategy (3 retries with exponential backoff)

**Implementation:**
```python
_session = requests.Session()
_adapter = HTTPAdapter(
    max_retries=Retry(total=3, backoff_factor=0.5, ...),
    pool_connections=10,
    pool_maxsize=20
)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)
```

**Why:**
- Reuses TCP connections instead of creating new ones per request
- Reduces overhead of SSL/TLS handshakes (especially for HTTPS)
- Automatic retries on transient failures (5xx errors, 429)

**Performance Impact:** ~50-100ms per API call (especially noticeable on repeated calls to the same backend)

---

## 4. Persistent Checkpointing (SQLite/PostgreSQL)

**What Changed:**
- Replaced `InMemorySaver()` with persistent storage (defaults to SQLite)
- Supports PostgreSQL for production deployments
- Configurable via `CHECKPOINTER_URL` or `CHECKPOINTER_DB` env vars

**How to Use:**

*SQLite (default):*
```bash
CHECKPOINTER_DB=./checkpoints.db
```

*PostgreSQL (production):*
```bash
CHECKPOINTER_URL=postgresql://user:password@localhost:5432/langgraph
```

**Why:**
- Conversation state survives app restarts
- Enables multi-turn conversations to continue across sessions
- Production-ready for stateful agent deployments
- Can replay or rewind conversations for debugging

---

## 5. Setup Script (`setup.sh`)

**What It Does:**
1. Creates `.env` from `.env.example` (with user prompt to edit)
2. Installs dependencies via `uv` or `pip`
3. Initializes Python virtual environment (if needed)
4. Sets up SQLite checkpoints database

**Usage:**
```bash
./setup.sh
# Follow prompts to enter credentials
python main.py
```

**What Gets Created:**
- `.env` with all required API keys and configuration
- `.venv/` (Python virtual environment)
- `checkpoints.db` (SQLite conversation history)

---

## Configuration Summary

### Environment Variables

| Variable | Purpose | Default | Example |
|----------|---------|---------|---------|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Required | `sk-...` |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | Required | `https://xxx.openai.azure.com/` |
| `AZURE_OPENAI_DEPLOYMENT` | Azure deployment name | Required | `gpt-5-mini` |
| `AGENTPHONE_API_KEY` | AgentPhone integration | Optional | |
| `AGENTPHONE_WEBHOOK_SECRET` | AgentPhone webhook security | Optional | |
| `CHECKPOINTER_URL` | PostgreSQL connection string | SQLite | `postgresql://...` |
| `CHECKPOINTER_DB` | SQLite database path | `./checkpoints.db` | |
| `HTTP_POOL_CONNECTIONS` | HTTP pool base size | 10 | |
| `HTTP_POOL_MAXSIZE` | HTTP pool max size | 20 | |

---

## Performance Summary

| Improvement | Latency Saved | Notes |
|-------------|---------------|-------|
| Remove closure LLM | 500-1000ms | Eliminates one LLM call per turn |
| Connection pooling | 50-100ms | Per API request |
| Persistent checkpointing | N/A | Enables state recovery |
**Total potential latency reduction:** 550-1100ms per turn

---

## Next Steps

1. **Run setup:**
   ```bash
   ./setup.sh
   ```

2. **Test with conversation:**
   ```bash
   python main.py
   ```

3. **(Optional) Use PostgreSQL:**
   - Set `CHECKPOINTER_URL` for production deployments
   - Enables distributed agent sessions
