# Lincoln Airport Agent

AI airport navigation assistant with phone support over Twilio voice + SMS.

## Setup

1. Install dependencies

```bash
uv sync
```

2. Configure environment

```bash
cp .env.example .env
```

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
