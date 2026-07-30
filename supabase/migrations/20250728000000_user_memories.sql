-- Per-user memory store.
-- One row per durable fact about a user ("prefers halal food", "flies Southwest").
-- Plain text, keyed by the hashed user id. No embeddings / vector search — the
-- LangGraph recall_memory node uses the chat model to pick relevant memories.

CREATE TABLE IF NOT EXISTS user_memories (
    id           uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id_hash text        NOT NULL,
    content      text        NOT NULL,
    category     text        DEFAULT 'other',   -- dietary|payment|accessibility|travel|preference|personal|other
    metadata     jsonb       DEFAULT '{}',
    created_at   timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS user_memories_user_idx ON user_memories (user_id_hash);
