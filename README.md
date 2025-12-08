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
          serviceAccountName: litellm-sync-sa
          containers:
          - name: sync
            image: gcr.io/YOUR-PROJECT/litellm-pg2bq-sync:latest
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
            - name: GOOGLE_APPLICATION_CREDENTIALS
              value: "/var/secrets/google/key.json"
            volumeMounts:
            - name: config
              mountPath: /app/config
              readOnly: true
            - name: gcp-key
              mountPath: /var/secrets/google
              readOnly: true
          volumes:
          - name: config
            configMap:
              name: litellm-sync-config
          - name: gcp-key
            secret:
              secretName: gcp-service-account-key
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: litellm-sync-sa
  namespace: default
  annotations:
    iam.gke.io/gcp-service-account: litellm-sync@YOUR-PROJECT.iam.gserviceaccount.com
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: litellm-sync-config
  namespace: default
data:
  config.json: |
    {
      "postgres_connection_string": "postgresql+psycopg2://postgres:PASSWORD@postgres-host:5432/usecase_storage_db",
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
      "max_retries": 2,
      "batch_size": 1000
    }
---
apiVersion: v1
kind: Secret
metadata:
  name: gcp-service-account-key
  namespace: default
type: Opaque
data:
  # Base64 encoded service account JSON
  # Create with: kubectl create secret generic gcp-service-account-key --from-file=key.json=path/to/service-account.json
  key.json: BASE64_ENCODED_SERVICE_ACCOUNT_JSON