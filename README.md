{
  "postgres_connection_string": "postgresql+psycopg2://postgres:GenAISept2025@localhost:5431/usecase_storage_db",
  "bigquery_project": "hbl-dev-gcp-gen-ai-prj-spk-5a",
  "bigquery_dataset": "dev_litellm_spend_logs_dataset",
  "bigquery_service_account_path": "secrets/gcp-service-account.json",
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
