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

## Endpoints

- `POST /twilio/voice` incoming call webhook
- `POST /twilio/sms` incoming SMS webhook
- `WS /twilio/stream` Twilio media stream socket
- `POST /gate-change` outbound gate-change notification
- `GET /health` health check

## Notes

- Twilio signature validation is active when `TWILIO_AUTH_TOKEN` is set.
- For local development, webhook URLs must match your live tunnel URL.
