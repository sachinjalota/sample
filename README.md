=================================================================== FAILURES ======================================================================================
__________________________________________________________________________ test_delete_collection_success ___________________________________________________________________________

client = <starlette.testclient.TestClient object at 0x124f2d490>, delete_collection_payload = {'collection_uid': 'test-collection-uuid'}
valid_headers = {'x-base-api-key': 'test-api-key', 'x-session-id': 'test-session'}, mocker = <pytest_mock.plugin.MockerFixture object at 0x1252c0e50>

    def test_delete_collection_success(client, delete_collection_payload, valid_headers, mocker):
        fake_session = MagicMock()
        fake_cm = MagicMock()
        fake_cm.__enter__.return_value = fake_session
        fake_cm.__exit__.return_value = None
        mocker.patch(
            "src.api.routers.collection_router.create_session_platform",
            return_value=fake_cm,
        )
    
        fake_session.execute.return_value.scalar_one_or_none.return_value = MagicMock(uuid="test-collection-uuid")
    
        mocker.patch(
            "src.repository.document_repository.DocumentRepository.check_table_exists",
            return_value=True
        )
        mocker.patch(
            "src.repository.document_repository.DocumentRepository.delete_collection",
            return_value=None
        )
    
        resp = client.request("DELETE", "/v1/api/delete_collection",
                              json=delete_collection_payload, headers=valid_headers
                              )
>       assert resp.status_code == 200, resp.text
E       AssertionError: {"error":"Invalid or unauthorized API key"}
E       assert 401 == 200
E        +  where 401 = <Response [401 Unauthorized]>.status_code

tests/unit/api/router/test_collection_router.py:145: AssertionError
------------------------------------------------------------------------------- Captured stdout call --------------------------------------------------------------------------------
[ERROR] [2025-06-12 14:04:47,166] [src.main] [137] Http error: 401: Invalid or unauthorized API key
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
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 75, in delete_collection
    await validate_collection_access(header_information.x_base_api_key, request.collection_uid)
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/deps.py", line 177, in validate_collection_access
    raise HTTPException(status_code=401, detail="Invalid or unauthorized API key")
fastapi.exceptions.HTTPException: 401: Invalid or unauthorized API key
--------------------------------------------------------------------------------- Captured log call ---------------------------------------------------------------------------------
ERROR    src.main:main.py:137 Http error: 401: Invalid or unauthorized API key
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
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 75, in delete_collection
    await validate_collection_access(header_information.x_base_api_key, request.collection_uid)
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/deps.py", line 177, in validate_collection_access
    raise HTTPException(status_code=401, detail="Invalid or unauthorized API key")
fastapi.exceptions.HTTPException: 401: Invalid or unauthorized API key
______________________________________________________________________ test_delete_collection_table_not_found _______________________________________________________________________

client = <starlette.testclient.TestClient object at 0x124fbc110>, delete_collection_payload = {'collection_uid': 'test-collection-uuid'}
valid_headers = {'x-base-api-key': 'test-api-key', 'x-session-id': 'test-session'}, mocker = <pytest_mock.plugin.MockerFixture object at 0x125401e10>

    def test_delete_collection_table_not_found(client, delete_collection_payload, valid_headers, mocker):
        fake_session = MagicMock()
        fake_cm = MagicMock()
        fake_cm.__enter__.return_value = fake_session
        fake_cm.__exit__.return_value = None
        mocker.patch(
            "src.api.routers.collection_router.create_session_platform",
            return_value=fake_cm,
        )
    
        fake_session.execute.return_value.scalar_one_or_none.return_value = MagicMock(uuid="test-collection-uuid")
    
        mocker.patch(
            "src.repository.document_repository.DocumentRepository.check_table_exists",
            return_value=False
        )
    
        resp = client.request("DELETE", "/v1/api/delete_collection",
                              json=delete_collection_payload, headers=valid_headers
                              )
>       assert resp.status_code == 404
E       assert 401 == 404
E        +  where 401 = <Response [401 Unauthorized]>.status_code

tests/unit/api/router/test_collection_router.py:171: AssertionError
------------------------------------------------------------------------------- Captured stdout call --------------------------------------------------------------------------------
[ERROR] [2025-06-12 14:04:47,309] [src.main] [137] Http error: 401: Invalid or unauthorized API key
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
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 75, in delete_collection
    await validate_collection_access(header_information.x_base_api_key, request.collection_uid)
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/deps.py", line 177, in validate_collection_access
    raise HTTPException(status_code=401, detail="Invalid or unauthorized API key")
fastapi.exceptions.HTTPException: 401: Invalid or unauthorized API key
--------------------------------------------------------------------------------- Captured log call ---------------------------------------------------------------------------------
ERROR    src.main:main.py:137 Http error: 401: Invalid or unauthorized API key
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
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 75, in delete_collection
    await validate_collection_access(header_information.x_base_api_key, request.collection_uid)
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/deps.py", line 177, in validate_collection_access
    raise HTTPException(status_code=401, detail="Invalid or unauthorized API key")
fastapi.exceptions.HTTPException: 401: Invalid or unauthorized API key
================================================================================= warnings summary ==================================================================================
src/db/base.py:3
  /Users/epfn119476/Documents/HDFC/genai_platform_services/src/db/base.py:3: MovedIn20Warning: The ``declarative_base()`` function is now available as sqlalchemy.orm.declarative_base(). (deprecated since: 2.0) (Background on SQLAlchemy 2.0 at: https://sqlalche.me/e/b8d9)
    BaseDBA = declarative_base()

.venv/lib/python3.11/site-packages/pydantic/_internal/_config.py:295
.venv/lib/python3.11/site-packages/pydantic/_internal/_config.py:295
.venv/lib/python3.11/site-packages/pydantic/_internal/_config.py:295
  /Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/pydantic/_internal/_config.py:295: PydanticDeprecatedSince20: Support for class-based `config` is deprecated, use ConfigDict instead. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.10/migration/
    warnings.warn(DEPRECATION_MESSAGE, DeprecationWarning)

<frozen importlib._bootstrap>:241
<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

<frozen importlib._bootstrap>:241
<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute

<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
