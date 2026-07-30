-- Retire the unused vector search on user_memories.
-- Recall is done by the chat model in the recall_memory node, not by embeddings,
-- so the embedding column, its index, and the match RPC are dead. Run this on
-- any database created from the earlier vector version of the table.

DROP FUNCTION IF EXISTS match_user_memories(vector, text, int, float);
DROP INDEX IF EXISTS user_memories_embedding_idx;
ALTER TABLE IF EXISTS user_memories DROP COLUMN IF EXISTS embedding;
