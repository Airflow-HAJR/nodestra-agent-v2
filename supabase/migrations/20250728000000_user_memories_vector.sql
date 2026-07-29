-- Per-user semantic memory store.
-- Each row is one durable fact about a user ("prefers halal food",
-- "uses a wheelchair", "flies Southwest"). Unlike vector_documents
-- (partitioned by airport, holds flight/store data), this is keyed by the
-- hashed user id and holds free-form personal facts the agent decides are
-- worth remembering across calls.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS user_memories (
    id           uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id_hash text        NOT NULL,
    content      text        NOT NULL,          -- concise 3rd-person fact
    category     text        DEFAULT 'other',   -- dietary|payment|accessibility|travel|preference|personal|other
    metadata     jsonb       DEFAULT '{}',
    embedding    vector(1536),
    created_at   timestamptz DEFAULT now()
);

-- Bulk load at session start filters by user; keep it fast.
CREATE INDEX IF NOT EXISTS user_memories_user_idx
    ON user_memories (user_id_hash);

-- Semantic recall.
CREATE INDEX IF NOT EXISTS user_memories_embedding_idx
    ON user_memories
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Optional server-side semantic search (the app normally recalls from an
-- in-session cache, but this RPC is here for one-off / debugging lookups).
CREATE OR REPLACE FUNCTION match_user_memories(
    query_embedding      vector(1536),
    match_user           text,
    match_count          int   DEFAULT 5,
    similarity_threshold float DEFAULT 0.0
)
RETURNS TABLE (
    id         uuid,
    content    text,
    category   text,
    metadata   jsonb,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.content,
        m.category,
        m.metadata,
        1 - (m.embedding <=> query_embedding) AS similarity
    FROM user_memories m
    WHERE
        m.user_id_hash = match_user
        AND 1 - (m.embedding <=> query_embedding) >= similarity_threshold
    ORDER BY m.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
