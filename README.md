[hdfcbank@genai-hdfc-jump-vm genai-agent-mesh]$ kubectl create job --from=cronjob/litellm-pg2bq-sync litellm-pg2bq-sync-manual -n services-dev
job.batch/litellm-pg2bq-sync-manual created
[hdfcbank@genai-hdfc-jump-vm genai-agent-mesh]$ kubectl get pods -n services-dev | grep litellm-pg2bq-sync-manual
litellm-pg2bq-sync-manual-b77hb                                0/2     Init:0/1           0                  1s
[hdfcbank@genai-hdfc-jump-vm genai-agent-mesh]$ kubectl logs -n services-dev -f litellm-pg2bq-sync-manual-b77hb 
<frozen runpy>:128: RuntimeWarning: 'litellm_sync.sync' found in sys.modules after import of package 'litellm_sync', but prior to execution of 'litellm_sync.sync'; this may result in unpredictable behaviour
2025-12-15 17:49:49,984 - __main__ - ERROR - Failed to connect to PostgreSQL: (psycopg2.OperationalError) connection to server at "10.216.89.14", port 5432 failed: Connection refused
	Is the server running on that host and accepting TCP/IP connections?

(Background on this error at: https://sqlalche.me/e/20/e3q8)
2025-12-15 17:49:49,984 - __main__ - ERROR - Fatal error: (psycopg2.OperationalError) connection to server at "10.216.89.14", port 5432 failed: Connection refused
	Is the server running on that host and accepting TCP/IP connections?

(Background on this error at: https://sqlalche.me/e/20/e3q8)
Traceback (most recent call last):
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/engine/base.py", line 143, in __init__
    self._dbapi_connection = engine.raw_connection()
                             ^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/engine/base.py", line 3309, in raw_connection
    return self.pool.connect()
           ^^^^^^^^^^^^^^^^^^^
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/pool/base.py", line 447, in connect
    return _ConnectionFairy._checkout(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/pool/base.py", line 1264, in _checkout
    fairy = _ConnectionRecord.checkout(pool)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/pool/base.py", line 711, in checkout
    rec = pool._do_get()
          ^^^^^^^^^^^^^^
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/pool/impl.py", line 177, in _do_get
    with util.safe_reraise():
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/util/langhelpers.py", line 224, in __exit__
    raise exc_value.with_traceback(exc_tb)
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/pool/impl.py", line 175, in _do_get
    return self._create_connection()
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/pool/base.py", line 388, in _create_connection
    return _ConnectionRecord(self)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/pool/base.py", line 673, in __init__
    self.__connect()
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/pool/base.py", line 899, in __connect
    with util.safe_reraise():
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/util/langhelpers.py", line 224, in __exit__
    raise exc_value.with_traceback(exc_tb)
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/pool/base.py", line 895, in __connect
    self.dbapi_connection = connection = pool._invoke_creator(self)
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/engine/create.py", line 661, in connect
    return dialect.connect(*cargs, **cparams)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/engine/default.py", line 630, in connect
    return self.loaded_dbapi.connect(*cargs, **cparams)  # type: ignore[no-any-return]  # NOQA: E501
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/app-root/lib64/python3.11/site-packages/psycopg2/__init__.py", line 122, in connect
    conn = _connect(dsn, connection_factory=connection_factory, **kwasync)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
psycopg2.OperationalError: connection to server at "10.216.89.14", port 5432 failed: Connection refused
	Is the server running on that host and accepting TCP/IP connections?


The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/app/src/litellm_sync/sync.py", line 445, in main
    sync = LiteLLMBigQuerySync(config_path="config/config.json")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/src/litellm_sync/sync.py", line 82, in __init__
    self.litellm_engine = self._connect_postgres(self.config['litellm_db'])
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/src/litellm_sync/sync.py", line 106, in _connect_postgres
    with engine.connect() as conn:
         ^^^^^^^^^^^^^^^^
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/engine/base.py", line 3285, in connect
    return self._connection_cls(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/engine/base.py", line 145, in __init__
    Connection._handle_dbapi_exception_noconnection(
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/engine/base.py", line 2448, in _handle_dbapi_exception_noconnection
    raise sqlalchemy_exception.with_traceback(exc_info[2]) from e
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/engine/base.py", line 143, in __init__
    self._dbapi_connection = engine.raw_connection()
                             ^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/engine/base.py", line 3309, in raw_connection
    return self.pool.connect()
           ^^^^^^^^^^^^^^^^^^^
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/pool/base.py", line 447, in connect
    return _ConnectionFairy._checkout(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/pool/base.py", line 1264, in _checkout
    fairy = _ConnectionRecord.checkout(pool)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/pool/base.py", line 711, in checkout
    rec = pool._do_get()
          ^^^^^^^^^^^^^^
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/pool/impl.py", line 177, in _do_get
    with util.safe_reraise():
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/util/langhelpers.py", line 224, in __exit__
    raise exc_value.with_traceback(exc_tb)
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/pool/impl.py", line 175, in _do_get
    return self._create_connection()
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/pool/base.py", line 388, in _create_connection
    return _ConnectionRecord(self)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/pool/base.py", line 673, in __init__
    self.__connect()
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/pool/base.py", line 899, in __connect
    with util.safe_reraise():
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/util/langhelpers.py", line 224, in __exit__
    raise exc_value.with_traceback(exc_tb)
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/pool/base.py", line 895, in __connect
    self.dbapi_connection = connection = pool._invoke_creator(self)
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/engine/create.py", line 661, in connect
    return dialect.connect(*cargs, **cparams)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/app-root/lib64/python3.11/site-packages/sqlalchemy/engine/default.py", line 630, in connect
    return self.loaded_dbapi.connect(*cargs, **cparams)  # type: ignore[no-any-return]  # NOQA: E501
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/app-root/lib64/python3.11/site-packages/psycopg2/__init__.py", line 122, in connect
    conn = _connect(dsn, connection_factory=connection_factory, **kwasync)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "10.216.89.14", port 5432 failed: Connection refused
	Is the server running on that host and accepting TCP/IP connections?

(Background on this error at: https://sqlalche.me/e/20/e3q8)








This is my K8 manifest file that gets kubectl apply
#apiVersion: external-secrets.io/v1beta1
#kind: ExternalSecret
#metadata:
#  name: litellm-postgres-credentials
#  namespace: services-dev
#spec:
#  refreshInterval: 1h
#  secretStoreRef:
#    name: gcpsm-secret-store
#    kind: ClusterSecretStore
#  target:
#    name: litellm-postgres-credentials
#    creationPolicy: Owner
#  data:
#  - secretKey: connection_string
#    remoteRef:
#      key: projects/626624654237/secrets/LITELLM_SYNC_DEV_DATABASE_URL/versions/2
---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: litellm-pg2bq-sync
  namespace: services-dev
spec:
  # Run at 12:05 AM IST daily (18:35 UTC, since IST = UTC+5:30)
  schedule: "35 18 * * *"
  timeZone: "UTC"
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      backoffLimit: 1
      template:
        metadata:
          labels:
            app: litellm-pg2bq-sync
        spec:
          restartPolicy: OnFailure
          serviceAccount: ksa-genai
          containers:
          - name: sync
            image: asia-south1-docker.pkg.dev/hbl-dev-gcp-gen-ai-prj-spk-5a/gen-ai-docker-repo/litellm-pg2bq-sync:v20251212_1
            imagePullPolicy: Always
            resources:
              requests:
                memory: "512Mi"
                cpu: "250m"
              limits:
                memory: "1Gi"
                cpu: "500m"
            env:
            - name: TZ
              value: "Asia/Kolkata"
#            - name: POSTGRES_CONNECTION_STRING
#              valueFrom:
#                secretKeyRef:
#                  name: litellm-postgres-credentials
#                  key: connection_string
            volumeMounts:
            - name: config
              mountPath: /app/config
              readOnly: true
          volumes:
          - name: config
            configMap:
              name: litellm-sync-config
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: litellm-sync-config
  namespace: services-dev
data:
  config.json: |
    {
      "postgres_connection_string": "postgresql+psycopg2://postgres:GenAISept2025@10.216.89.14:5432/",
      "litellm_db": "litellm_db",
      "platform_meta_db": "platform_meta_db",
      "bigquery_project": "hbl-dev-gcp-gen-ai-prj-spk-5a",
      "bigquery_dataset": "dev_litellm_spend_logs_dataset",
      "tables": [
        "LiteLLM_OrganizationTable",
        "LiteLLM_TeamTable",
        "LiteLLM_DailyTeamSpend"
      ],
      "timestamp_columns": [
        "created_at",
        "updated_at"
      ],
      "primary_keys": {
        "LiteLLM_OrganizationTable": "organization_id",
        "LiteLLM_TeamTable": "team_id",
        "LiteLLM_DailyTeamSpend": "id"
      },
      "max_retries": 1,
      "batch_size": 2000
    }
