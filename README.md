apiVersion: batch/v1
kind: CronJob
metadata:
  name: litellm-pg2bq-sync
  namespace: default
spec:
  # Run at 12:05 AM IST daily (18:35 UTC, since IST = UTC+5:30)
  schedule: "35 18 * * *"
  timeZone: "UTC"
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      backoffLimit: 2
      template:
        metadata:
          labels:
            app: litellm-pg2bq-sync
        spec:
          restartPolicy: OnFailure
          containers:
          - name: sync
            image: your-registry/litellm-pg2bq-sync:latest
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
            volumeMounts:
            - name: config
              mountPath: /app/config
              readOnly: true
            - name: secrets
              mountPath: /app/secrets
              readOnly: true
          volumes:
          - name: config
            configMap:
              name: litellm-sync-config
          - name: secrets
            secret:
              secretName: litellm-sync-secrets
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: litellm-sync-config
  namespace: default
data:
  config.json: |
    {
      "postgres_connection_string": "postgresql+psycopg2://postgres:PASSWORD_PLACEHOLDER@postgres-service:5432/usecase_storage_db",
      "bigquery_project": "hbl-dev-gcp-gen-ai-prj-spk-5a",
      "bigquery_dataset": "dev_litellm_spend_logs_dataset",
      "bigquery_service_account_path": "/app/secrets/gcp-service-account.json",
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
      "max_retries": 2,
      "batch_size": 1000
    }
---
apiVersion: v1
kind: Secret
metadata:
  name: litellm-sync-secrets
  namespace: default
type: Opaque
stringData:
  gcp-service-account.json: |
    {
      "type": "service_account",
      "project_id": "your-project",
      "private_key_id": "key-id",
      "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
      "client_email": "service-account@project.iam.gserviceaccount.com",
      "client_id": "123456789",
      "auth_uri": "https://accounts.google.com/o/oauth2/auth",
      "token_uri": "https://oauth2.googleapis.com/token",
      "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
      "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/..."
    }
