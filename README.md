ALL CODE IS DISTRIBUTED ON THIS ORGANIZATION's 3 REPOSITORIES. See the other two repositories in this organization for the hydration layer built with MOSS and the custom map builder we created to provide an indoor map of OAK airport to the agent.

AI airport navigation assistant with phone support over Twilio voice + SMS.

Using conversational AI, users can call a phone number and receive real-time navigation, flight updates, accessibility assistance, and personalized recommendations.

1. Install dependencies

DeepIndoors can remember that a user:

2. Configure environment

Tech Stack:
- LangGraph
- Moss.dev
- AgentPhone
- Supermemory

Set these for Twilio local hosting:
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_PHONE_NUMBER` (your purchased Twilio number, E.164 format like `+14155551234`)
- `DEEPGRAM_API_KEY`
- `ELEVENLABS_API_KEY`
- `SERVER_BASE_URL` (public tunnel URL while running locally)

## Text Test UI (no voice)

A minimal browser chat for testing the agent + semantic memory system without
Twilio or audio.

1. Apply the memory migration once in the Supabase SQL editor:
   `supabase/migrations/20250728000000_user_memories_vector.sql`
2. For semantic recall, set `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` to a deployed
   embedding model. Without it, memory falls back to substring matching.
3. Run the UI:

```bash
uv run uvicorn webchat:app --reload --port 8100
```

Open http://localhost:8100. The left panel shows what the agent has remembered
about the current user id. State a fact ("I only eat halal food"), start a new
conversation, and ask for recommendations — it recalls from the per-user vector
store. Memory is loaded once per session and refreshed in-process on write.

## Run Locally With Twilio

1. Start the app:

```bash
uv run uvicorn server:app --reload --port 8000
```

2. Start a tunnel and set `SERVER_BASE_URL` to that URL:

```bash
ngrok http 8000
```

3. In Twilio Console for your phone number:
- Voice webhook: `https://<your-tunnel>/twilio/voice` (HTTP `POST`)
- Messaging webhook: `https://<your-tunnel>/twilio/sms` (HTTP `POST`)

4. Call or text your Twilio number.

## Accounts and memory

Memory is keyed by whoever the server can actually identify:

| Who | Key | What survives the session |
| --- | --- | --- |
| Phone caller | their number (carrier-verified) | everything durable they say |
| Signed-in web user | `sub:<supabase user id>` | everything durable they say |
| Web guest | a random per-device id | nothing |

Guests are not a degraded experience — the agent still extracts their
preferences and acts on them for the rest of the conversation. Those facts just
never reach the database. That's what makes signing in worth offering, and it
means a shared kiosk doesn't quietly build a profile of everyone who walks up.

The browser proves an identity by sending its Supabase access token in the
`config` websocket message; `agent/auth.py` verifies it against Supabase's auth
API (never by decoding the JWT locally) and the result is what sets
`persist_memory` on the graph. An unverified token is treated as a guest. If a
user signs in mid-conversation, the socket stays up and the graph reloads their
memories in place — and anything learned while they were a guest is written to
the account rather than dropped.

Account-facing endpoints, all bearer-authenticated and all scoped to the
caller's own rows:

- `GET /account/me` profile + everything remembered about them
- `DELETE /account/memories/{id}` forget one fact
- `DELETE /account/memories` forget everything

## Endpoints

- `POST /twilio/voice` incoming call webhook
- `POST /twilio/sms` incoming SMS webhook
- `WS /twilio/stream` Twilio media stream socket
- `WS /web/stream` browser voice socket
- `POST /gate-change` outbound gate-change notification
- `GET /health` health check

## Notes

- Twilio signature validation is active when `TWILIO_AUTH_TOKEN` is set.
- For local development, webhook URLs must match your live tunnel URL.
