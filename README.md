apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "litellm-pg2bq-sync.fullname" . }}-config
  namespace: {{ .Values.namespace }}
  labels:
    {{- include "litellm-pg2bq-sync.labels" . | nindent 4 }}
data:
  config.json: |
    {
      "postgres_connection_string": "WILL_BE_OVERRIDDEN_BY_ENV",
      "litellm_db": {{ .Values.config.litellm_db | quote }},
      "platform_meta_db": {{ .Values.config.platform_meta_db | quote }},
      "bigquery_project": {{ .Values.config.bigquery_project | quote }},
      "bigquery_dataset": {{ .Values.config.bigquery_dataset | quote }},
      "tables": {{ .Values.config.tables | toJson }},
      "timestamp_columns": {{ .Values.config.timestamp_columns | toJson }},
      "primary_keys": {{ .Values.config.primary_keys | toJson }},
      "max_retries": {{ .Values.config.max_retries }},
      "batch_size": {{ .Values.config.batch_size }}
    }
