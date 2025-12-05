(litellm-pg2bq-sync) (base) epfn119476@25C-LTP-H-39281 litellm-pg2bq % python -m litellm_sync.sync
<frozen runpy>:128: RuntimeWarning: 'litellm_sync.sync' found in sys.modules after import of package 'litellm_sync', but prior to execution of 'litellm_sync.sync'; this may result in unpredictable behaviour
2025-12-05 11:29:19,106 - __main__ - INFO - Connected to PostgreSQL
2025-12-05 11:29:19,847 - __main__ - INFO - Connected to BigQuery using Application Default Credentials
2025-12-05 11:29:20,027 - __main__ - INFO - Sync state table ensured
2025-12-05 11:29:20,027 - __main__ - INFO - Starting sync for 3 tables
2025-12-05 11:29:20,027 - __main__ - INFO - Starting sync for table: LiteLLM_OrganizationTable
2025-12-05 11:29:20,849 - __main__ - INFO - Creating BigQuery table: LiteLLM_OrganizationTable
2025-12-05 11:29:21,179 - __main__ - ERROR - Error syncing table LiteLLM_OrganizationTable: 400 POST https://bigquery.googleapis.com/bigquery/v2/projects/hbl-dev-gcp-gen-ai-prj-spk-5a/datasets/dev_litellm_spend_logs_dataset/tables?prettyPrint=false: Table with field based partitioning must have a schema.
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/src/litellm_sync/sync.py", line 360, in sync_table
    self._ensure_bq_table(table_name)
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/src/litellm_sync/sync.py", line 233, in _ensure_bq_table
    table = self.bq_client.create_table(table)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/bigquery/client.py", line 827, in create_table
    api_response = self._call_api(
                   ^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/bigquery/client.py", line 861, in _call_api
    return call()
           ^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 294, in retry_wrapped_func
    return retry_target(
           ^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 156, in retry_target
    next_sleep = _retry_error_helper(
                 ^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_base.py", line 214, in _retry_error_helper
    raise final_exc from source_exc
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 147, in retry_target
    result = target()
             ^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/_http/__init__.py", line 494, in api_request
    raise exceptions.from_http_response(response)
google.api_core.exceptions.BadRequest: 400 POST https://bigquery.googleapis.com/bigquery/v2/projects/hbl-dev-gcp-gen-ai-prj-spk-5a/datasets/dev_litellm_spend_logs_dataset/tables?prettyPrint=false: Table with field based partitioning must have a schema.
2025-12-05 11:29:21,189 - __main__ - INFO - Retrying LiteLLM_OrganizationTable (attempt 1/2)
2025-12-05 11:29:26,190 - __main__ - INFO - Starting sync for table: LiteLLM_OrganizationTable
2025-12-05 11:29:26,524 - __main__ - INFO - Creating BigQuery table: LiteLLM_OrganizationTable
2025-12-05 11:29:26,660 - __main__ - ERROR - Error syncing table LiteLLM_OrganizationTable: 400 POST https://bigquery.googleapis.com/bigquery/v2/projects/hbl-dev-gcp-gen-ai-prj-spk-5a/datasets/dev_litellm_spend_logs_dataset/tables?prettyPrint=false: Table with field based partitioning must have a schema.
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/src/litellm_sync/sync.py", line 360, in sync_table
    self._ensure_bq_table(table_name)
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/src/litellm_sync/sync.py", line 233, in _ensure_bq_table
    table = self.bq_client.create_table(table)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/bigquery/client.py", line 827, in create_table
    api_response = self._call_api(
                   ^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/bigquery/client.py", line 861, in _call_api
    return call()
           ^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 294, in retry_wrapped_func
    return retry_target(
           ^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 156, in retry_target
    next_sleep = _retry_error_helper(
                 ^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_base.py", line 214, in _retry_error_helper
    raise final_exc from source_exc
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 147, in retry_target
    result = target()
             ^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/_http/__init__.py", line 494, in api_request
    raise exceptions.from_http_response(response)
google.api_core.exceptions.BadRequest: 400 POST https://bigquery.googleapis.com/bigquery/v2/projects/hbl-dev-gcp-gen-ai-prj-spk-5a/datasets/dev_litellm_spend_logs_dataset/tables?prettyPrint=false: Table with field based partitioning must have a schema.
2025-12-05 11:29:26,662 - __main__ - INFO - Retrying LiteLLM_OrganizationTable (attempt 2/2)
2025-12-05 11:29:31,667 - __main__ - INFO - Starting sync for table: LiteLLM_OrganizationTable
2025-12-05 11:29:31,884 - __main__ - INFO - Creating BigQuery table: LiteLLM_OrganizationTable
2025-12-05 11:29:32,137 - __main__ - ERROR - Error syncing table LiteLLM_OrganizationTable: 400 POST https://bigquery.googleapis.com/bigquery/v2/projects/hbl-dev-gcp-gen-ai-prj-spk-5a/datasets/dev_litellm_spend_logs_dataset/tables?prettyPrint=false: Table with field based partitioning must have a schema.
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/src/litellm_sync/sync.py", line 360, in sync_table
    self._ensure_bq_table(table_name)
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/src/litellm_sync/sync.py", line 233, in _ensure_bq_table
    table = self.bq_client.create_table(table)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/bigquery/client.py", line 827, in create_table
    api_response = self._call_api(
                   ^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/bigquery/client.py", line 861, in _call_api
    return call()
           ^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 294, in retry_wrapped_func
    return retry_target(
           ^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 156, in retry_target
    next_sleep = _retry_error_helper(
                 ^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_base.py", line 214, in _retry_error_helper
    raise final_exc from source_exc
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 147, in retry_target
    result = target()
             ^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/_http/__init__.py", line 494, in api_request
    raise exceptions.from_http_response(response)
google.api_core.exceptions.BadRequest: 400 POST https://bigquery.googleapis.com/bigquery/v2/projects/hbl-dev-gcp-gen-ai-prj-spk-5a/datasets/dev_litellm_spend_logs_dataset/tables?prettyPrint=false: Table with field based partitioning must have a schema.
2025-12-05 11:29:32,284 - __main__ - INFO - Starting sync for table: LiteLLM_TeamTable
2025-12-05 11:29:32,619 - __main__ - INFO - Creating BigQuery table: LiteLLM_TeamTable
2025-12-05 11:29:32,766 - __main__ - ERROR - Error syncing table LiteLLM_TeamTable: 400 POST https://bigquery.googleapis.com/bigquery/v2/projects/hbl-dev-gcp-gen-ai-prj-spk-5a/datasets/dev_litellm_spend_logs_dataset/tables?prettyPrint=false: Table with field based partitioning must have a schema.
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/src/litellm_sync/sync.py", line 360, in sync_table
    self._ensure_bq_table(table_name)
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/src/litellm_sync/sync.py", line 233, in _ensure_bq_table
    table = self.bq_client.create_table(table)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/bigquery/client.py", line 827, in create_table
    api_response = self._call_api(
                   ^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/bigquery/client.py", line 861, in _call_api
    return call()
           ^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 294, in retry_wrapped_func
    return retry_target(
           ^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 156, in retry_target
    next_sleep = _retry_error_helper(
                 ^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_base.py", line 214, in _retry_error_helper
    raise final_exc from source_exc
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 147, in retry_target
    result = target()
             ^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/_http/__init__.py", line 494, in api_request
    raise exceptions.from_http_response(response)
google.api_core.exceptions.BadRequest: 400 POST https://bigquery.googleapis.com/bigquery/v2/projects/hbl-dev-gcp-gen-ai-prj-spk-5a/datasets/dev_litellm_spend_logs_dataset/tables?prettyPrint=false: Table with field based partitioning must have a schema.
2025-12-05 11:29:32,768 - __main__ - INFO - Retrying LiteLLM_TeamTable (attempt 1/2)
2025-12-05 11:29:37,769 - __main__ - INFO - Starting sync for table: LiteLLM_TeamTable
2025-12-05 11:29:37,995 - __main__ - INFO - Creating BigQuery table: LiteLLM_TeamTable
2025-12-05 11:29:38,278 - __main__ - ERROR - Error syncing table LiteLLM_TeamTable: 400 POST https://bigquery.googleapis.com/bigquery/v2/projects/hbl-dev-gcp-gen-ai-prj-spk-5a/datasets/dev_litellm_spend_logs_dataset/tables?prettyPrint=false: Table with field based partitioning must have a schema.
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/src/litellm_sync/sync.py", line 360, in sync_table
    self._ensure_bq_table(table_name)
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/src/litellm_sync/sync.py", line 233, in _ensure_bq_table
    table = self.bq_client.create_table(table)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/bigquery/client.py", line 827, in create_table
    api_response = self._call_api(
                   ^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/bigquery/client.py", line 861, in _call_api
    return call()
           ^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 294, in retry_wrapped_func
    return retry_target(
           ^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 156, in retry_target
    next_sleep = _retry_error_helper(
                 ^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_base.py", line 214, in _retry_error_helper
    raise final_exc from source_exc
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 147, in retry_target
    result = target()
             ^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/_http/__init__.py", line 494, in api_request
    raise exceptions.from_http_response(response)
google.api_core.exceptions.BadRequest: 400 POST https://bigquery.googleapis.com/bigquery/v2/projects/hbl-dev-gcp-gen-ai-prj-spk-5a/datasets/dev_litellm_spend_logs_dataset/tables?prettyPrint=false: Table with field based partitioning must have a schema.
2025-12-05 11:29:38,280 - __main__ - INFO - Retrying LiteLLM_TeamTable (attempt 2/2)
2025-12-05 11:29:43,285 - __main__ - INFO - Starting sync for table: LiteLLM_TeamTable
2025-12-05 11:29:43,647 - __main__ - INFO - Creating BigQuery table: LiteLLM_TeamTable
2025-12-05 11:29:43,791 - __main__ - ERROR - Error syncing table LiteLLM_TeamTable: 400 POST https://bigquery.googleapis.com/bigquery/v2/projects/hbl-dev-gcp-gen-ai-prj-spk-5a/datasets/dev_litellm_spend_logs_dataset/tables?prettyPrint=false: Table with field based partitioning must have a schema.
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/src/litellm_sync/sync.py", line 360, in sync_table
    self._ensure_bq_table(table_name)
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/src/litellm_sync/sync.py", line 233, in _ensure_bq_table
    table = self.bq_client.create_table(table)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/bigquery/client.py", line 827, in create_table
    api_response = self._call_api(
                   ^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/bigquery/client.py", line 861, in _call_api
    return call()
           ^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 294, in retry_wrapped_func
    return retry_target(
           ^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 156, in retry_target
    next_sleep = _retry_error_helper(
                 ^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_base.py", line 214, in _retry_error_helper
    raise final_exc from source_exc
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 147, in retry_target
    result = target()
             ^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/_http/__init__.py", line 494, in api_request
    raise exceptions.from_http_response(response)
google.api_core.exceptions.BadRequest: 400 POST https://bigquery.googleapis.com/bigquery/v2/projects/hbl-dev-gcp-gen-ai-prj-spk-5a/datasets/dev_litellm_spend_logs_dataset/tables?prettyPrint=false: Table with field based partitioning must have a schema.
2025-12-05 11:29:43,959 - __main__ - INFO - Starting sync for table: LiteLLM_DailyTeamSpend
2025-12-05 11:29:44,193 - __main__ - INFO - Creating BigQuery table: LiteLLM_DailyTeamSpend
2025-12-05 11:29:44,335 - __main__ - ERROR - Error syncing table LiteLLM_DailyTeamSpend: 400 POST https://bigquery.googleapis.com/bigquery/v2/projects/hbl-dev-gcp-gen-ai-prj-spk-5a/datasets/dev_litellm_spend_logs_dataset/tables?prettyPrint=false: Table with field based partitioning must have a schema.
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/src/litellm_sync/sync.py", line 360, in sync_table
    self._ensure_bq_table(table_name)
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/src/litellm_sync/sync.py", line 233, in _ensure_bq_table
    table = self.bq_client.create_table(table)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/bigquery/client.py", line 827, in create_table
    api_response = self._call_api(
                   ^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/bigquery/client.py", line 861, in _call_api
    return call()
           ^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 294, in retry_wrapped_func
    return retry_target(
           ^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 156, in retry_target
    next_sleep = _retry_error_helper(
                 ^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_base.py", line 214, in _retry_error_helper
    raise final_exc from source_exc
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 147, in retry_target
    result = target()
             ^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/_http/__init__.py", line 494, in api_request
    raise exceptions.from_http_response(response)
google.api_core.exceptions.BadRequest: 400 POST https://bigquery.googleapis.com/bigquery/v2/projects/hbl-dev-gcp-gen-ai-prj-spk-5a/datasets/dev_litellm_spend_logs_dataset/tables?prettyPrint=false: Table with field based partitioning must have a schema.
2025-12-05 11:29:44,338 - __main__ - INFO - Retrying LiteLLM_DailyTeamSpend (attempt 1/2)
2025-12-05 11:29:49,342 - __main__ - INFO - Starting sync for table: LiteLLM_DailyTeamSpend
2025-12-05 11:29:49,626 - __main__ - INFO - Creating BigQuery table: LiteLLM_DailyTeamSpend
2025-12-05 11:29:49,772 - __main__ - ERROR - Error syncing table LiteLLM_DailyTeamSpend: 400 POST https://bigquery.googleapis.com/bigquery/v2/projects/hbl-dev-gcp-gen-ai-prj-spk-5a/datasets/dev_litellm_spend_logs_dataset/tables?prettyPrint=false: Table with field based partitioning must have a schema.
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/src/litellm_sync/sync.py", line 360, in sync_table
    self._ensure_bq_table(table_name)
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/src/litellm_sync/sync.py", line 233, in _ensure_bq_table
    table = self.bq_client.create_table(table)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/bigquery/client.py", line 827, in create_table
    api_response = self._call_api(
                   ^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/bigquery/client.py", line 861, in _call_api
    return call()
           ^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 294, in retry_wrapped_func
    return retry_target(
           ^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 156, in retry_target
    next_sleep = _retry_error_helper(
                 ^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_base.py", line 214, in _retry_error_helper
    raise final_exc from source_exc
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 147, in retry_target
    result = target()
             ^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/_http/__init__.py", line 494, in api_request
    raise exceptions.from_http_response(response)
google.api_core.exceptions.BadRequest: 400 POST https://bigquery.googleapis.com/bigquery/v2/projects/hbl-dev-gcp-gen-ai-prj-spk-5a/datasets/dev_litellm_spend_logs_dataset/tables?prettyPrint=false: Table with field based partitioning must have a schema.
2025-12-05 11:29:49,775 - __main__ - INFO - Retrying LiteLLM_DailyTeamSpend (attempt 2/2)
2025-12-05 11:29:54,779 - __main__ - INFO - Starting sync for table: LiteLLM_DailyTeamSpend
2025-12-05 11:29:54,986 - __main__ - INFO - Creating BigQuery table: LiteLLM_DailyTeamSpend
2025-12-05 11:29:55,222 - __main__ - ERROR - Error syncing table LiteLLM_DailyTeamSpend: 400 POST https://bigquery.googleapis.com/bigquery/v2/projects/hbl-dev-gcp-gen-ai-prj-spk-5a/datasets/dev_litellm_spend_logs_dataset/tables?prettyPrint=false: Table with field based partitioning must have a schema.
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/src/litellm_sync/sync.py", line 360, in sync_table
    self._ensure_bq_table(table_name)
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/src/litellm_sync/sync.py", line 233, in _ensure_bq_table
    table = self.bq_client.create_table(table)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/bigquery/client.py", line 827, in create_table
    api_response = self._call_api(
                   ^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/bigquery/client.py", line 861, in _call_api
    return call()
           ^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 294, in retry_wrapped_func
    return retry_target(
           ^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 156, in retry_target
    next_sleep = _retry_error_helper(
                 ^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_base.py", line 214, in _retry_error_helper
    raise final_exc from source_exc
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/api_core/retry/retry_unary.py", line 147, in retry_target
    result = target()
             ^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/cloud/_http/__init__.py", line 494, in api_request
    raise exceptions.from_http_response(response)
google.api_core.exceptions.BadRequest: 400 POST https://bigquery.googleapis.com/bigquery/v2/projects/hbl-dev-gcp-gen-ai-prj-spk-5a/datasets/dev_litellm_spend_logs_dataset/tables?prettyPrint=false: Table with field based partitioning must have a schema.
2025-12-05 11:29:55,366 - __main__ - INFO - Sync completed: 0 successful, 3 failed, 0 total records
2025-12-05 11:29:55,366 - __main__ - INFO - Sync results: {
  "total_tables": 3,
  "successful": 0,
  "failed": 3,
  "total_records_synced": 0,
  "timestamp": "2025-12-05T05:59:55.366239+00:00",
  "table_results": {
    "LiteLLM_OrganizationTable": {
      "status": "FAILED",
      "error": "400 POST https://bigquery.googleapis.com/bigquery/v2/projects/hbl-dev-gcp-gen-ai-prj-spk-5a/datasets/dev_litellm_spend_logs_dataset/tables?prettyPrint=false: Table with field based partitioning must have a schema.",
      "retry_count": 2
    },
    "LiteLLM_TeamTable": {
      "status": "FAILED",
      "error": "400 POST https://bigquery.googleapis.com/bigquery/v2/projects/hbl-dev-gcp-gen-ai-prj-spk-5a/datasets/dev_litellm_spend_logs_dataset/tables?prettyPrint=false: Table with field based partitioning must have a schema.",
      "retry_count": 2
    },
    "LiteLLM_DailyTeamSpend": {
      "status": "FAILED",
      "error": "400 POST https://bigquery.googleapis.com/bigquery/v2/projects/hbl-dev-gcp-gen-ai-prj-spk-5a/datasets/dev_litellm_spend_logs_dataset/tables?prettyPrint=false: Table with field based partitioning must have a schema.",
      "retry_count": 2
    }
  }
}
2025-12-05 11:29:55,374 - __main__ - INFO - Connections closed
(litellm-pg2bq-sync) (base) epfn119476@25C-LTP-H-39281 litellm-pg2bq % 

