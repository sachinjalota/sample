# LiteLLM PostgreSQL to BigQuery Sync

Incremental sync tool that transfers data from LiteLLM PostgreSQL tables to BigQuery, handling schema changes automatically.

## Features

- **Incremental Sync**: Only syncs new/updated records based on `created_at` and `updated_at` timestamps
- **State Management**: Tracks last sync timestamp in PostgreSQL state table
- **Auto Schema Detection**: Automatically creates and updates BigQuery tables based on PostgreSQL schema
- **Upsert Logic**: Handles updates using MERGE statements
- **Retry Mechanism**: Retries failed tables twice before marking as failed
- **Dynamic Configuration**: Easy to add new tables via config file
- **Kubernetes Ready**: Runs as CronJob at 12:05 AM IST daily

## Project Structure

```
litellm-pg2bq-sync/
├── src/
│   └── litellm_sync/
│       ├── __init__.py
│       └── sync.py              # Main sync logic
├── config/
│   └── config.json.template     # Configuration template
├── secrets/
│   └── gcp-service-account.json # GCP service account key (gitignored)
├── Dockerfile                   # Container image
├── k8s-cronjob.yaml            # Kubernetes CronJob manifest
├── pyproject.toml              # Python project configuration
├── README.md
└── .gitignore
```

## Prerequisites

- Python 3.9+
- PostgreSQL with LiteLLM tables
- Google Cloud Platform project with BigQuery
- GCP Service Account with BigQuery permissions
- Kubernetes cluster (for production deployment)

## Setup

### 1. Clone and Install

```bash
# Clone repository
git clone <repo-url>
cd litellm-pg2bq-sync

# Install in development mode
pip install -e .
```

### 2. Configuration

Create `config/config.json` from template:

```bash
cp config/config.json.template config/config.json
```

Edit `config/config.json` with your settings:

```json
{
  "postgres_connection_string": "postgresql+psycopg2://user:password@host:port/database",
  "bigquery_project": "your-gcp-project-id",
  "bigquery_dataset": "your_dataset_name",
  "bigquery_service_account_path": "secrets/gcp-service-account.json",
  "tables": [
    "LiteLLM_OrganizationTable",
    "LiteLLM_TeamTable",
    "LiteLLM_DailyTeamSpend"
  ],
  "timestamp_columns": ["created_at", "updated_at"],
  "primary_keys": {
    "LiteLLM_OrganizationTable": "organization_id",
    "LiteLLM_TeamTable": "team_id",
    "LiteLLM_DailyTeamSpend": "id"
  },
  "max_retries": 2,
  "batch_size": 1000
}
```

### 3. GCP Service Account

1. Create a service account in GCP Console
2. Grant roles:
   - BigQuery Data Editor
   - BigQuery Job User
3. Download JSON key and save to `secrets/gcp-service-account.json`

### 4. SSH Tunnel (Development)

For local development with SSH tunnel to PostgreSQL:

```bash
ssh -L 5431:localhost:5432 user@your-postgres-server -N
```

## Usage

### Run Locally

```bash
# Run sync
python -m litellm_sync.sync

# Or using entry point
litellm-sync
```

### Build Docker Image

```bash
docker build -t litellm-pg2bq-sync:latest .
```

### Test Docker Locally

```bash
docker run --rm \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/secrets:/app/secrets \
  litellm-pg2bq-sync:latest
```

### Deploy to Kubernetes

1. **Push image to registry**:
```bash
docker tag litellm-pg2bq-sync:latest your-registry/litellm-pg2bq-sync:latest
docker push your-registry/litellm-pg2bq-sync:latest
```

2. **Update k8s-cronjob.yaml**:
   - Replace `your-registry` with your container registry
   - Update ConfigMap with your configuration
   - Add GCP service account JSON to Secret

3. **Create Kubernetes Secret for PostgreSQL password**:
```bash
kubectl create secret generic postgres-credentials \
  --from-literal=password='GenAISept2025'
```

4. **Deploy**:
```bash
kubectl apply -f k8s-cronjob.yaml
```

5. **Verify**:
```bash
# Check CronJob
kubectl get cronjobs

# View job runs
kubectl get jobs

# Check logs
kubectl logs -l app=litellm-pg2bq-sync
```

## Adding New Tables

To add new tables for sync:

1. Edit `config/config.json`
2. Add table name to `tables` array
3. Add primary key mapping in `primary_keys` object
4. Redeploy (for Kubernetes) or restart sync

Example:
```json
{
  "tables": [
    "LiteLLM_OrganizationTable",
    "LiteLLM_TeamTable",
    "LiteLLM_DailyTeamSpend",
    "LiteLLM_NewTable"  // New table
  ],
  "primary_keys": {
    "LiteLLM_NewTable": "id"  // Add primary key
  }
}
```

## How It Works

### Sync Process

1. **Initialize**: Connects to PostgreSQL and BigQuery
2. **State Check**: Reads last sync timestamp from `litellm_sync_state` table
3. **Schema Sync**: Ensures BigQuery table exists with correct schema
4. **Incremental Fetch**: Queries PostgreSQL for records where `created_at > last_sync` OR `updated_at > last_sync`
5. **Upsert**: Uses MERGE statement to insert new records or update existing ones
6. **State Update**: Records successful sync timestamp
7. **Retry**: Retries failed tables up to 2 times

### State Table

A `litellm_sync_state` table is automatically created in PostgreSQL:

```sql
CREATE TABLE litellm_sync_state (
    table_name VARCHAR(255) PRIMARY KEY,
    last_sync_timestamp TIMESTAMP,
    last_sync_status VARCHAR(50),
    last_sync_record_count INTEGER,
    updated_at TIMESTAMP
);
```

### Schema Changes

The tool automatically handles:
- **New columns**: Added to BigQuery table
- **Type changes**: Manual intervention required
- **Column removals**: Old columns remain in BigQuery

## Monitoring

### Check Sync Status

Query the state table in PostgreSQL:

```sql
SELECT * FROM litellm_sync_state ORDER BY updated_at DESC;
```

### View Logs

Kubernetes:
```bash
kubectl logs -l app=litellm-pg2bq-sync --tail=100
```

Local:
```bash
# Logs go to stdout
```

### Verify BigQuery Data

```sql
SELECT 
  COUNT(*) as record_count,
  MAX(updated_at) as latest_record
FROM `project.dataset.table_name`;
```

## Troubleshooting

### Connection Issues

**PostgreSQL Connection Failed**:
- Check connection string in config
- Verify SSH tunnel is active (for local dev)
- Check firewall rules

**BigQuery Connection Failed**:
- Verify service account JSON path
- Check service account permissions
- Ensure BigQuery API is enabled

### Sync Issues

**No Records Syncing**:
- Check `litellm_sync_state` table for last sync timestamp
- Verify timestamp columns exist in source table
- Check if there are actually new records

**Schema Mismatch**:
- Review BigQuery table schema
- Check PostgreSQL column types
- Manual ALTER TABLE may be needed for type changes

**Job Failures**:
```bash
# View failed job logs
kubectl logs <pod-name>

# Check job status
kubectl describe job <job-name>
```

## Configuration Reference

| Parameter | Type | Description |
|-----------|------|-------------|
| `postgres_connection_string` | string | PostgreSQL connection URL |
| `bigquery_project` | string | GCP project ID |
| `bigquery_dataset` | string | BigQuery dataset name |
| `bigquery_service_account_path` | string | Path to GCP service account JSON |
| `tables` | array | List of table names to sync |
| `timestamp_columns` | array | Columns used for incremental sync |
| `primary_keys` | object | Primary key column for each table |
| `max_retries` | integer | Number of retry attempts for failed tables |
| `batch_size` | integer | Number of records per batch |

## Security Best Practices

1. **Never commit secrets**: Use `.gitignore` for `secrets/` and `config/config.json`
2. **Use Kubernetes Secrets**: Store credentials in K8s secrets, not ConfigMaps
3. **Limit Service Account**: Grant minimum required BigQuery permissions
4. **Rotate Credentials**: Regularly rotate PostgreSQL and GCP credentials
5. **Network Security**: Use private GKE clusters and Cloud SQL Private IP

## Performance Tuning

- **Batch Size**: Adjust `batch_size` for large tables (default: 1000)
- **Partitioning**: BigQuery tables are automatically partitioned by `updated_at`
- **Indexes**: Ensure PostgreSQL has indexes on timestamp columns
- **Resources**: Adjust Kubernetes resource limits based on data volume

## License

MIT License

## Support

For issues or questions:
1. Check logs for error messages
2. Review troubleshooting section
3. Open an issue in the repository
