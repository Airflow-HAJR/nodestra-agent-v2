# Running the memory + RAG + navigation demo

The beat this demo is built around: a traveler asks what there is to eat, the
agent shows real options on the map, then remembers — out loud, with
attribution — that on their last trip they were checking whether places were
halal, checks all three against retrieved venue write-ups, and walks them to
the one that qualifies.

Three systems on screen at once: **memory** (what it knows from last time),
**retrieval** (where the halal answer comes from), **navigation** (the map and
the checkpoints).

---

## 1. Seed the traveler's history

Memory fills itself from conversation, so on a clean database there is no
"last time". This writes one:

```bash
cd nodestra-agent-v2
uv run python scripts/seed_demo_memories.py            # or: .venv/bin/python
uv run python scripts/seed_demo_memories.py --show     # check what's stored
```

Five facts land under user id `+15105550142`, each stamped with where and when
it was learned — that stamp is what lets the agent say "last time you came
through OAK…" instead of just knowing things.

## 2. Rehearse it headlessly (30 seconds, no browser)

```bash
uv run python scripts/demo_run.py
```

Prints each turn's tool calls, map actions, and reply. If a beat doesn't land
here it won't land on stage. Useful flags: `--guest` (no stored memories — the
same question, no halal check, which is a good contrast to show), `--turn "..."`
to drive it yourself.

## 3. Run it for real

Backend:

```bash
cd nodestra-agent-v2
DEMO_USER_ID=+15105550142 DEMO_START_POI=security-4RXW \
  uv run uvicorn server:app --port 8000
```

`DEMO_USER_ID` serves every web session as that seeded traveler with durable
memory on, whether or not anyone signs in — so you don't have to do a Google
sign-in on stage. **It is an authentication bypass** (anyone opening the page
gets those memories), so keep it out of any deployed environment. The server
logs a warning at startup whenever it's set. `DEMO_START_POI` drops the
traveler at Terminal 1 Security on the first turn, so the demo opens on
"what's nearby?" instead of the agent asking where they are.

Frontend:

```bash
cd nodestra-agent-ui
npm run dev            # http://localhost:5173/oakland/
```

## 4. The script

| You say | What to point at |
| --- | --- |
| "Hey, I'm really hungry — what's nearby?" | Three green pins drop on the map with walking times. The agent names each in one sentence. |
| *(same turn)* | It says "last time you came through OAK you were checking whether places were halal, so I checked all three" — then gives a verdict for **each**, failures included, and recommends the one that passes. That verdict comes from retrieval, not from the model's opinion of a restaurant chain. |
| "Yeah, take me there." | The map switches to the route: origin, every named stop, destination, ETA, and a "Made it to __" button for the first checkpoint. |
| "Made it to Lake Merritt Essentials." | Advances one checkpoint, re-highlights the next. Confirm checkpoints **in order** — saying you're somewhere further along advances one step and then points you back at where you already are. |

Optional closers:

- **"What do you remember about me?"** — it reads back the stored facts.
- **"Actually I'm vegetarian now."** — watch the terminal: `[memory: saved
  (dietary, db) — ...]`. It's written in the background, after the reply, so it
  costs the turn nothing. Ask for food again and the new fact is already in play.
- Run the same first question with `DEMO_USER_ID` unset (a guest) to show the
  contrast: same three places, no halal check, nothing remembered afterward.

## Before you present

- Dismiss the "Save your preferences" sign-in popup once ("Not now") — it fires
  ~6s after load and covers the top-right corner. It re-arms after 12 hours.
- Load the page once and ask one throwaway question first: the embedding index
  builds on the first run (a few seconds, then cached to `data/poi_index.json`),
  and the browser asks for location permission.
- Check the terminal shows `LLM_PROVIDER=azure` is in effect — the default is
  `openai`, which needs an `OPENAI_API_KEY` this machine doesn't have.

## What's actually wired

- **Memory** — `agent/vector_memory.py` (Supabase, one row per fact),
  loaded once per session by the `init` node in `agent/graph.py`, dumped into
  the system prompt by `agent/prompts.py`, and written back in a background
  thread after each turn. Facts carry `metadata` (source / airport / date),
  which the prompt renders as "(learned: …)" so the agent can attribute what it
  knows rather than just asserting it.
- **Retrieval** — `agent/poi_rag.py` over `data/poi_knowledge.json`. Chunks are
  embedded once (cached, keyed by a hash of the corpus + model) and searched by
  cosine similarity; with no embedding model configured it falls back to keyword
  overlap rather than failing. Two tools use it: `suggest_places` (what's
  nearby, ranked by relevance *and* walking distance, pinned on the map) and
  `lookup_poi_info` (checks a question against specific venues — one retrieval
  per venue, so every candidate comes back with its own evidence instead of the
  strongest match drowning out the rest).
- **The nudge that makes it reliable** — `suggest_places` reads the traveler's
  own memories and returns the constraining ones back to the agent alongside the
  shortlist, with instructions to check them before recommending. Rather than
  hoping the model connects "eats halal" to three restaurants it was just
  handed, the shortlist carries the constraint.
- **Map** — `show_map_options` (new) pins several candidates with no route line
  between them, since they're alternatives rather than stops; the existing
  `show_map_trajectory` / `advance_checkpoint` handle the walk.

> **The venue write-ups in `data/poi_knowledge.json` are invented for this
> demo** — the halal certifications, hours, and prices are not sourced from OAK
> or the restaurants. The POI ids are real, so the retrieved text pins and routes
> correctly, but replace the corpus with licensed or scraped data before this is
> in front of a real traveler.
