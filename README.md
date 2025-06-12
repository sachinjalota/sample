2025-06-12 19:34:14,467 INFO sqlalchemy.engine.Engine BEGIN (implicit)
2025-06-12 19:34:14,468 INFO sqlalchemy.engine.Engine SELECT embedding_models.model_name, embedding_models.dimensions 
FROM embedding_models 
WHERE embedding_models.model_name = %(model_name_1)s
2025-06-12 19:34:14,468 INFO sqlalchemy.engine.Engine [cached since 24.41s ago] {'model_name_1': 'BAAI/bge-m3'}
2025-06-12 19:34:14,538 INFO sqlalchemy.engine.Engine COMMIT
[ERROR] [2025-06-12 19:34:14,572] [src.api.routers.collection_router] [63] Error creating collection.
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 36, in create_collection
    dims = await check_embedding_model(entry.model)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/utility/collection_helpers.py", line 15, in check_embedding_model
    return int(row.dimensions)
               ^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/sqlalchemy/orm/attributes.py", line 566, in __get__
    return self.impl.get(state, dict_)  # type: ignore[no-any-return]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/sqlalchemy/orm/attributes.py", line 1086, in get
    value = self._fire_loader_callables(state, key, passive)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/sqlalchemy/orm/attributes.py", line 1116, in _fire_loader_callables
    return state._load_expired(state, passive)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/sqlalchemy/orm/state.py", line 803, in _load_expired
    self.manager.expired_attribute_loader(self, toload, passive)
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/sqlalchemy/orm/loading.py", line 1603, in load_scalar_attributes
    raise orm_exc.DetachedInstanceError(
sqlalchemy.orm.exc.DetachedInstanceError: Instance <EmbeddingModels at 0x115ef1710> is not bound to a Session; attribute refresh operation cannot proceed (Background on this error at: https://sqlalche.me/e/20/bhk3)
[ERROR] [2025-06-12 19:34:14,573] [src.main] [140] Http error: 500: Instance <EmbeddingModels at 0x115ef1710> is not bound to a Session; attribute refresh operation cannot proceed (Background on this error at: https://sqlalche.me/e/20/bhk3)
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 36, in create_collection
    dims = await check_embedding_model(entry.model)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/utility/collection_helpers.py", line 15, in check_embedding_model
    return int(row.dimensions)
               ^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/sqlalchemy/orm/attributes.py", line 566, in __get__
    return self.impl.get(state, dict_)  # type: ignore[no-any-return]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/sqlalchemy/orm/attributes.py", line 1086, in get
    value = self._fire_loader_callables(state, key, passive)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/sqlalchemy/orm/attributes.py", line 1116, in _fire_loader_callables
    return state._load_expired(state, passive)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/sqlalchemy/orm/state.py", line 803, in _load_expired
    self.manager.expired_attribute_loader(self, toload, passive)
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/sqlalchemy/orm/loading.py", line 1603, in load_scalar_attributes
    raise orm_exc.DetachedInstanceError(
sqlalchemy.orm.exc.DetachedInstanceError: Instance <EmbeddingModels at 0x115ef1710> is not bound to a Session; attribute refresh operation cannot proceed (Background on this error at: https://sqlalche.me/e/20/bhk3)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/starlette/routing.py", line 73, in app
    response = await f(request)
               ^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/fastapi/routing.py", line 301, in app
    raw_response = await run_endpoint_function(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/fastapi/routing.py", line 212, in run_endpoint_function
    return await dependant.call(**values)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 64, in create_collection
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
fastapi.exceptions.HTTPException: 500: Instance <EmbeddingModels at 0x115ef1710> is not bound to a Session; attribute refresh operation cannot proceed (Background on this error at: https://sqlalche.me/e/20/bhk3)
INFO:     127.0.0.1:65107 - "POST /DEV/platform-services/v1/api/create_collection HTTP/1.1" 500 Internal Server Error


