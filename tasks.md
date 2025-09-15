1. Align collection & index APIs with OpenAI reference
   Update the public collection‑related endpoints (/collection, /collection/data, /collection/create, /collection/delete, /index, /search, /delete_index, /collection/delete_by_ids) . Add proper OpenAPI metadata and reuse existing service logic.
2. OpenAI Vector‑Store client SDK
   Create a thin wrapper (src/integrations/openai_vectorstore_sdk.py). This SDK will be used by the routers and can also be reused by other services (e.g., RAG).
3. Update existing OpenAI SDK to call the new Vector‑Store SDK for indexing/search
4. Add/Update Pydantic models	Align request/response models (CreateCollection, DeleteCollection, IndexingPayload, SearchRequest, DeleteRequest, DeleteByIdsRequest) with the OpenAI schema (e.g., rename fields, add optional metadata, name, expires_after).
5. Implement ElasticSeach
6. Add unit tests for new endpoints & back‑ends	Extend tests/unit/api with tests that cover the new OpenAI‑compatible routes, the ElasticSearch backend and the new SDK.
