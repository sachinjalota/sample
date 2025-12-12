apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: litellm-postgres-credentials
  namespace: services-dev
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: gcpsm-secret-store
    kind: ClusterSecretStore
  target:
    name: litellm-postgres-credentials
    creationPolicy: Owner
  data:
  - secretKey: connection_string
    remoteRef:
      key: projects/626624654237/secrets/LITELLM_SYNC_DEV_DATABASE_URL/versions/1
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
            image: asia-south1-docker.pkg.dev/hbl-dev-gcp-gen-ai-prj-spk-5a/genai-docker-virtual-repo/litellm-pg2bq-sync:latest
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
            - name: POSTGRES_CONNECTION_STRING
              valueFrom:
                secretKeyRef:
                  name: litellm-postgres-credentials
                  key: connection_string
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
      "postgres_connection_string": "WILL_BE_OVERRIDDEN_BY_ENV",
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
