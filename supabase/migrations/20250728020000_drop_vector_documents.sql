-- Remove the flight/store catalog vector store entirely.
--
-- vector_documents (and its partitions vector_documents_oak / _default) held
-- OAK store + flight sample data, searched by the search_store_info /
-- search_flight_info tools. Those tools and agent/local_search.py have been
-- deleted, so the table and its RPC are dead.

DROP TABLE IF EXISTS vector_documents CASCADE;
DROP FUNCTION IF EXISTS match_documents(vector, text, text, int);
