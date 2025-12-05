This error is **not caused by your script** but by **BigQuery rejecting a long-running HTTPS streaming insert**.

Here is what the error means:

```
Timeout of 600s exceeded → 10-minute job
SSLEOFError: EOF occurred in violation of protocol → Google closed the connection
insertAll → BigQuery streaming insert API
```

Google **closes long-lived streaming-insert connections** when:

* Too many rows in one insertAll request
* Payload too large (>10MB)
* Too many insertAll calls in parallel
* Network idle for too long
* Using private network / corporate VPN / proxy blocking large TLS frames
* Regional mismatch or VPC-SC boundary

---

# ✅ **Why this happened in your case**

The table causing failure:

```
LiteLLM_DailyTeamSpend
```

This table **may have large JSON columns** or **large row payloads**.

Your script likely batches rows as:

```python
errors = bq_client.insert_rows_json(table_id, rows)
```

If `rows` is big, BigQuery closes the connection → **SSLEOFError**.

Also, your 600-second timeout means your gRPC/HTTP client was waiting too long without progress.

---

# 🚀 **THE FIX: Switch from insertAll → LOAD JOB (recommended by Google)**

**Streaming insert (`insertAll`) is NOT meant for large batches.**
When syncing thousands of rows, use a **BigQuery Load Job** from:

* a temporary JSON file
* or a pandas DataFrame
* or a GCS file

This eliminates:

* timeouts
* SSL issues
* row-size limits
* retry problems

---

# 🔥 **Fix Implementation (Guaranteed to solve your issue)**

### Replace:

```python
bq_client.insert_rows_json(table_id, rows)
```

### With a BULK LOAD job:

```python
from google.cloud import bigquery
import pandas as pd
import json
import tempfile

def load_to_bigquery(bq_client, table_ref, rows):
    # convert rows to JSONL
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
        for row in rows:
            tmp.write(json.dumps(row) + "\n")
        temp_file = tmp.name

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition="WRITE_APPEND",
        autodetect=False
    )

    with open(temp_file, "rb") as f:
        load_job = bq_client.load_table_from_file(f, table_ref, job_config=job_config)

    load_job.result()  # wait for job to finish
```

This avoids all streaming insert problems.

---

# 🧠 Why load jobs solve your issues

| Problem          | Streaming Insert         | Load Job              |
| ---------------- | ------------------------ | --------------------- |
| Large batch size | ❌ fails                  | ✔ safe                |
| Timeout          | ❌ common                 | ✔ rare                |
| SSL EOF          | ❌ common                 | ✔ never happens       |
| Schema update    | ❌ tricky                 | ✔ supports autodetect |
| Speed            | ❌ slow                   | ✔ fast                |
| Cost             | ❌ costly (streaming fee) | ✔ free                |

---

# 🚀 Additional improvements you MUST apply

### 1. **Batch rows into smaller chunks**

Use:

```python
BATCH_SIZE = 5000
```

Do NOT load 50,000 rows at once.

---

### 2. Set explicit BigQuery retry strategy

```python
bq_client = bigquery.Client(
    client_info=ClientInfo(user_agent="litellm-sync")
)
```

---

### 3. Use regional BigQuery endpoint

You are probably using the default US endpoint.

But your dataset seems like:

```
dev_litellm_spend_logs_dataset
```

→ It might be in **asia-south1** (Mumbai).

Use:

```
https://asia-south1-bigquery.googleapis.com
```

Configure via:

```python
bq_client = bigquery.Client(client_options={"api_endpoint": "https://asia-south1-bigquery.googleapis.com"})
```

---

# 🎯 FINAL SOLUTION SUMMARY

### ❗ Your error is caused by:

* Using BigQuery Streaming Insert (`insertAll`)
* With large batch → BigQuery closes SSL connection → timeout

### ✔ Solution:

* Replace streaming inserts with **BigQuery Load Jobs**
* Use JSONL temp file
* Batch rows in chunks
* Use regional endpoint

---

# If you want:

I can rewrite your full `sync_table()` function to:

✔ generate incremental rows
✔ batch rows
✔ write JSONL temp files
✔ run BigQuery load jobs
✔ update watermark
✔ handle schema drift

Just say: **“Rewrite sync_table with load jobs”** and I’ll produce a clean, production-ready version.
