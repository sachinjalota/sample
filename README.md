k8s-cronjob.yaml
In my org service accounts are added in following format
serviceAccount: ksa-genai

also namespace would be 'services-dev'

my latest config looks like
{
  "postgres_connection_string": "postgresql+psycopg2://postgres:GenAISept2025@localhost:5431/",
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


In which i want my postgres_connection_string to be a google secret



Update the docker file as well, pull the base image from 
FROM asia-south1-docker.pkg.dev/hbl-dev-gcp-gen-ai-prj-spk-5a/genai-docker-virtual-repo/ubi9/python-311
