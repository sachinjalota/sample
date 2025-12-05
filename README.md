# Quick Setup Guide

## Step-by-Step Setup

### 1. Project Setup

```bash
# Create project directory
mkdir -p litellm-pg2bq-sync
cd litellm-pg2bq-sync

# Create directory structure
mkdir -p src/litellm_sync config secrets

# Create __init__.py
touch src/litellm_sync/__init__.py
```

### 2. Copy Files

Copy all the provided files into their respective directories:
- `sync.py` → `src/litellm_sync/`
- `__init__.py` → `src/litellm_sync/`
- `config.json.template` → `config/`
- `pyproject.toml` → root
- `Dockerfile` → root
- `k8s-cronjob.yaml` → root
- `README.md` → root
- `.gitignore` → root

### 3. Configure

```bash
# Copy config template
cp config/config.json.template config/config.json

# Edit configuration
nano config/config.json
```

Update these fields:
- `postgres_connection_string`: Your PostgreSQL connection
- `bigquery_project`: Your GCP project
- `bigquery_dataset`: Your BigQuery dataset
- `tables`: List of tables to sync
- `primary_keys`: Primary key for each table

### 4. GCP Service Account

1. Go to GCP Console → IAM & Admin → Service Accounts
2. Create new service account: `litellm-sync-sa`
3. Grant roles:
   - BigQuery Data Editor
   - BigQuery Job User
4. Create JSON key
5. Save as `secrets/gcp-service-account.json`

### 5. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

### 6. Test Locally

```bash
# If using SSH tunnel to PostgreSQL
ssh -L 5431:localhost:5432 user@db-server -N &

# Run sync
python -m litellm_sync.sync
```

### 7. Build Docker Image

```bash
docker build -t litellm-pg2bq-sync:latest .
```

### 8. Deploy to Kubernetes

```bash
# Update image in k8s-cronjob.yaml
# Replace: your-registry/litellm-pg2bq-sync:latest

# Push to registry
docker tag litellm-pg2bq-sync:latest gcr.io/YOUR-PROJECT/litellm-pg2bq-sync:latest
docker push gcr.io/YOUR-PROJECT/litellm-pg2bq-sync:latest

# Update k8s-cronjob.yaml with actual values:
# - Container image
# - PostgreSQL connection string
# - GCP service account JSON

# Deploy
kubectl apply -f k8s-cronjob.yaml

# Verify
kubectl get cronjobs
kubectl get jobs
```

### 9. Monitor

```bash
# Check CronJob schedule
kubectl describe cronjob litellm-pg2bq-sync

# View logs from last run
kubectl logs -l app=litellm-pg2bq-sync --tail=100

# Check sync state in PostgreSQL
psql -h localhost -p 5431 -U postgres -d usecase_storage_db \
  -c "SELECT * FROM litellm_sync_state;"
```

## Quick Verification

```bash
# 1. Check PostgreSQL state table
psql -c "SELECT * FROM litellm_sync_state ORDER BY updated_at DESC;"

# 2. Query BigQuery
bq query --use_legacy_sql=false \
  'SELECT COUNT(*) FROM `hbl-dev-gcp-gen-ai-prj-spk-5a.dev_litellm_spend_logs_dataset.LiteLLM_TeamTable`'

# 3. Check logs
kubectl logs -l app=litellm-pg2bq-sync --tail=50
```

## Troubleshooting Quick Fixes

**Can't connect to PostgreSQL**:
```bash
# Test connection
psql postgresql://postgres:GenAISept2025@localhost:5431/usecase_storage_db
```

**BigQuery auth fails**:
```bash
# Test service account
gcloud auth activate-service-account --key-file=secrets/gcp-service-account.json
bq ls --project_id=hbl-dev-gcp-gen-ai-prj-spk-5a
```

**No data syncing**:
```sql
-- Check if there's new data
SELECT COUNT(*), MAX(created_at), MAX(updated_at) 
FROM LiteLLM_TeamTable;

-- Check state table
SELECT * FROM litellm_sync_state;
```

**Pod not starting**:
```bash
# Check pod status
kubectl get pods -l app=litellm-pg2bq-sync

# View pod events
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name>
```

## CronJob Schedule Explanation

The CronJob runs at **12:05 AM IST** daily:
- IST = UTC + 5:30
- 12:05 AM IST = 6:35 PM UTC (previous day)
- Cron: `35 18 * * *` (UTC time)

To change schedule:
```yaml
# Run every 6 hours
schedule: "0 */6 * * *"

# Run at 2 AM IST (20:30 UTC previous day)
schedule: "30 20 * * *"
```

## Adding New Tables

1. Edit `config/config.json`:
```json
{
  "tables": [
    "LiteLLM_OrganizationTable",
    "LiteLLM_TeamTable",
    "LiteLLM_DailyTeamSpend",
    "LiteLLM_NewTable"  // Add here
  ],
  "primary_keys": {
    "LiteLLM_NewTable": "id"  // Add primary key
  }
}
```

2. For Kubernetes deployment:
```bash
# Update ConfigMap
kubectl edit configmap litellm-sync-config

# Or reapply
kubectl apply -f k8s-cronjob.yaml
```

3. Next scheduled run will automatically sync the new table

## Manual Trigger (Kubernetes)

```bash
# Create a one-time job from CronJob
kubectl create job --from=cronjob/litellm-pg2bq-sync manual-sync-$(date +%s)

# Watch job
kubectl get jobs -w

# View logs
kubectl logs -l app=litellm-pg2bq-sync --tail=100 -f
```
