image:
  registry: asia-south1-docker.pkg.dev
  repository: hbl-dev-gcp-gen-ai-prj-spk-5a/genai-docker-virtual-repo/litellm-pg2bq-sync
  tag: latest
  pullPolicy: Always

namespace: services-dev

serviceAccount:
  name: ksa-genai

cronjob:
  schedule: "35 18 * * *"  # 12:05 AM IST
  timeZone: "UTC"
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
  concurrencyPolicy: Forbid
  backoffLimit: 1

resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"

externalSecret:
  enabled: true
  name: litellm-postgres-credentials
  refreshInterval: 1h
  secretStoreRef:
    name: gcpsm-secret-store
    kind: ClusterSecretStore
  remoteSecretKey: projects/626624654237/secrets/LITELLM_SYNC_DEV_DATABASE_URL/versions/1

config:
  litellm_db: "litellm_db"
  platform_meta_db: "platform_meta_db"
  bigquery_project: "hbl-dev-gcp-gen-ai-prj-spk-5a"
  bigquery_dataset: "dev_litellm_spend_logs_dataset"
  tables:
    - LiteLLM_OrganizationTable
    - LiteLLM_TeamTable
    - LiteLLM_DailyTeamSpend
  timestamp_columns:
    - created_at
    - updated_at
  primary_keys:
    LiteLLM_OrganizationTable: organization_id
    LiteLLM_TeamTable: team_id
    LiteLLM_DailyTeamSpend: id
  max_retries: 1
  batch_size: 2000
