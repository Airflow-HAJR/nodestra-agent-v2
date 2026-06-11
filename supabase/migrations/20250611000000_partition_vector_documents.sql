-- Convert vector_documents to a LIST-partitioned table keyed by airport_id.
-- Each airport gets its own physical partition; the DEFAULT partition catches
-- any airport without a dedicated one.
-- Queries filtered by airport_id only touch one partition (partition pruning).

-- Step 1: preserve existing rows (drop the old index first — it keeps its name
--         after the rename and would collide when we recreate it on the new table)
DROP INDEX IF EXISTS vector_documents_embedding_idx;
ALTER TABLE IF EXISTS vector_documents RENAME TO vector_documents_legacy;

-- Step 2: partitioned parent — airport_id NOT NULL (required for partition key)
--         Primary key is composite so it includes the partition key.
CREATE TABLE vector_documents (
    id          text    NOT NULL,
    collection  text    NOT NULL,
    airport_id  text    NOT NULL,
    content     text    NOT NULL,
    metadata    jsonb   DEFAULT '{}',
    embedding   vector(1536),
    PRIMARY KEY (id, airport_id)
) PARTITION BY LIST (airport_id);

-- Step 3: named partitions — add one block per airport as you onboard them
CREATE TABLE vector_documents_oak     PARTITION OF vector_documents FOR VALUES IN ('OAK');
CREATE TABLE vector_documents_default PARTITION OF vector_documents DEFAULT;

-- Step 4: IVFFlat index on the parent; PostgreSQL propagates it to every
--         existing partition and to any partition added in the future.
CREATE INDEX vector_documents_embedding_idx
    ON vector_documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Step 5: migrate surviving rows (any row without airport_id is unroutable and dropped)
INSERT INTO vector_documents (id, collection, airport_id, content, metadata, embedding)
SELECT id, collection, airport_id, content, metadata, embedding
FROM   vector_documents_legacy
WHERE  airport_id IS NOT NULL
ON CONFLICT (id, airport_id) DO NOTHING;

DROP TABLE vector_documents_legacy;

-- match_documents RPC is unchanged — partition pruning fires automatically
-- when match_airport is provided.

-- To add a new airport later:
--   CREATE TABLE vector_documents_{lower_iata} PARTITION OF vector_documents
--       FOR VALUES IN ('{IATA}');
