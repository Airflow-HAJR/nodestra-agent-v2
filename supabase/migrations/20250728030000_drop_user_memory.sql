-- Remove the old singular user_memory table entirely.
--
-- It stored session-continuity fields (visit_count, last_flight, last_location)
-- and was the FK parent of calls/turns. All code that read or wrote it has been
-- deleted; user preferences now live in the user_memories (plural) vector table.
--
-- CASCADE drops the FK constraints on calls/turns that referenced user_memory.
-- The calls/turns tables themselves remain; their user_id_hash becomes a plain
-- column with no foreign key.

DROP TABLE IF EXISTS user_memory CASCADE;
