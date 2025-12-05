(litellm-pg2bq-sync) (base) epfn119476@25C-LTP-H-39281 litellm-pg2bq % cp config/config.json.template config/config.json
(litellm-pg2bq-sync) (base) epfn119476@25C-LTP-H-39281 litellm-pg2bq % python -m litellm_sync.sync
<frozen runpy>:128: RuntimeWarning: 'litellm_sync.sync' found in sys.modules after import of package 'litellm_sync', but prior to execution of 'litellm_sync.sync'; this may result in unpredictable behaviour
2025-12-05 10:43:42,504 - __main__ - ERROR - Failed to connect to PostgreSQL: invalid dsn: missing "=" after "postgresql+psycopg2://postgres:GenAISept2025@localhost:5431/litellm_db" in connection info string

2025-12-05 10:43:42,504 - __main__ - ERROR - Fatal error: invalid dsn: missing "=" after "postgresql+psycopg2://postgres:GenAISept2025@localhost:5431/litellm_db" in connection info string
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/src/litellm_sync/sync.py", line 442, in main
    sync = LiteLLMBigQuerySync(config_path="config/config.json")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/src/litellm_sync/sync.py", line 87, in __init__
    self.pg_conn = self._connect_postgres()
                   ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/src/litellm_sync/sync.py", line 104, in _connect_postgres
    conn = psycopg2.connect(self.config['postgres_connection_string'])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/psycopg2/__init__.py", line 121, in connect
    dsn = _ext.make_dsn(dsn, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/litellm-pg2bq/.venv/lib/python3.11/site-packages/psycopg2/extensions.py", line 145, in make_dsn
    parse_dsn(dsn)
psycopg2.ProgrammingError: invalid dsn: missing "=" after "postgresql+psycopg2://postgres:GenAISept2025@localhost:5431/litellm_db" in connection info string

(litellm-pg2bq-sync) (base) epfn119476@25C-LTP-H-39281 litellm-pg2bq % 

