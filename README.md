(litellm-pg2bq-sync) (base) epfn119476@25C-LTP-H-39281 litellm-pg2bq % python -m litellm_sync.sync
<frozen runpy>:128: RuntimeWarning: 'litellm_sync.sync' found in sys.modules after import of package 'litellm_sync', but prior to execution of 'litellm_sync.sync'; this may result in unpredictable behaviour
2025-12-05 11:01:10,378 - __main__ - INFO - Connected to PostgreSQL
2025-12-05 11:01:10,378 - __main__ - ERROR - Failed to connect to BigQuery: [Errno 2] No such file or directory: 'secrets/gcp-service-account.json'
2025-12-05 11:01:10,378 - __main__ - ERROR - Fatal error: [Errno 2] No such file or directory: 'secrets/gcp-service-account.json'
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/src/litellm_sync/sync.py", line 455, in main
    sync = LiteLLMBigQuerySync(config_path="config/config.json")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/src/litellm_sync/sync.py", line 96, in __init__
    self.bq_client = self._connect_bigquery()
                     ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/src/litellm_sync/sync.py", line 131, in _connect_bigquery
    credentials = service_account.Credentials.from_service_account_file(
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/oauth2/service_account.py", line 262, in from_service_account_file
    info, signer = _service_account_info.from_filename(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/google/auth/_service_account_info.py", line 78, in from_filename
    with io.open(filename, "r", encoding="utf-8") as json_file:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'secrets/gcp-service-account.json'
(litellm-pg2bq-sync) (base) epfn119476@25C-LTP-H-39281 litellm-pg2bq % 

