-- Enable pgvector (requires superuser or the extension to already be trusted)
CREATE EXTENSION IF NOT EXISTS vector;

-- Unified table for flight and store vector documents
CREATE TABLE IF NOT EXISTS vector_documents (
    id          text    PRIMARY KEY,
    collection  text    NOT NULL,    -- 'flights' | 'stores'
    airport_id  text,
    content     text    NOT NULL,
    metadata    jsonb   DEFAULT '{}',
    embedding   vector(1536)
);

CREATE INDEX IF NOT EXISTS vector_documents_embedding_idx
    ON vector_documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Semantic similarity search used by the agent
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding  vector(1536),
    match_collection text,
    match_airport    text    DEFAULT NULL,
    match_count      int     DEFAULT 5
)
RETURNS TABLE (
    id         text,
    content    text,
    metadata   jsonb,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        vd.id,
        vd.content,
        vd.metadata,
        1 - (vd.embedding <=> query_embedding) AS similarity
    FROM vector_documents vd
    WHERE
        vd.collection = match_collection
        AND (match_airport IS NULL OR vd.airport_id = match_airport)
    ORDER BY vd.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
