## 1. File Structure
/genai_platform_services/
├── src/
│   ├── celery_app.py          # NEW
│   ├── celery_tasks/          # NEW
│   │   ├── __init__.py
│   │   ├── file_processing.py
│   │   └── monitoring.py
│   └── ...
├── celery_worker.py            # NEW - Root level
├── init.sh                     # MODIFY
├── supervisor.conf             # NEW (if using single pod)
├── Dockerfile                  # MODIFY
├── pyproject.toml             # MODIFY
└── ...

## 2. Updated Dockerfile
FROM asia-south1-docker.pkg.dev/hbl-dev-gcp-gen-ai-prj-spk-5a/genai-docker-virtual-repo/ubi9/python-311

USER root

WORKDIR /app

ARG GCP_ARTIFACT_USERNAME=oauth2accesstoken
ARG GCP_ARTIFACT_PASSWORD
ARG PYPI_REPO_URL=asia-south1-python.pkg.dev/hbl-dev-gcp-gen-ai-prj-spk-5a/genai-python-virtual-repo/simple/

ENV APP_ROOT=/app/.venv \
    VIRTUAL_ENV=${APP_ROOT} \
    UV_PROJECT_ENVIRONMENT=${APP_ROOT} \
    UV_DEFAULT_INDEX=https://$GCP_ARTIFACT_USERNAME:$GCP_ARTIFACT_PASSWORD@$PYPI_REPO_URL \
    PATH="${APP_ROOT}/bin:$PATH" \
    APP_HOST="0.0.0.0" \
    APP_PORT=8080 \
    SERVICE_MODE="api"

RUN pip install --no-cache-dir --index-url https://$GCP_ARTIFACT_USERNAME:$GCP_ARTIFACT_PASSWORD@$PYPI_REPO_URL uv==0.5.28
RUN pip install --no-cache-dir --index-url https://$GCP_ARTIFACT_USERNAME:$GCP_ARTIFACT_PASSWORD@$PYPI_REPO_URL --upgrade pip setuptools wheel

COPY pyproject.toml .
COPY uv.lock .
COPY README.md .

# Install Dependencies (includes celery)
RUN uv sync --no-dev

COPY src ./src
COPY celery_worker.py .
COPY init.sh .
COPY supervisor.conf /etc/supervisor/conf.d/supervisor.conf

RUN chmod +x init.sh celery_worker.py

EXPOSE ${APP_PORT}

RUN chmod -R +x /app
RUN chown -R 1001:0 /app

USER 1001

CMD ["./init.sh"]

## 3. Updated init.sh
#!/bin/bash

set -e

# Check service mode from environment variable
SERVICE_MODE=${SERVICE_MODE:-api}

echo "Starting service in mode: $SERVICE_MODE"

case "$SERVICE_MODE" in
  api)
    echo "Starting FastAPI server..."
    exec uvicorn src.main:app --host ${APP_HOST} --port ${APP_PORT} --workers ${APP_WORKERS:--1}
    ;;
  
  celery)
    echo "Starting Celery worker..."
    exec python celery_worker.py
    ;;
  
  both)
    echo "Starting FastAPI + Celery with supervisor..."
    exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisor.conf
    ;;
  
  *)
    echo "Unknown SERVICE_MODE: $SERVICE_MODE"
    echo "Valid modes: api, celery, both"
    exit 1
    ;;
esac

## 4. supervisor.conf (for single-pod mode)
# supervisor.conf
[supervisord]
nodaemon=true
logfile=/tmp/supervisord.log
pidfile=/tmp/supervisord.pid
user=1001

[program:fastapi]
command=uvicorn src.main:app --host %(ENV_APP_HOST)s --port %(ENV_APP_PORT)s
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
user=1001

[program:celery_worker]
command=python celery_worker.py
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
stopwaitsecs=600
killasgroup=true
user=1001

## 5. celery_worker.py (Root Level)
#!/usr/bin/env python
"""
Celery worker entry point
Run with: python celery_worker.py
"""

from src.celery_app import celery_app
from src.config import get_settings
from src.logging_config import Logger

settings = get_settings()
logger = Logger.create_logger(__name__)

if __name__ == "__main__":
    logger.info("Starting Celery worker for GenAI Platform")
    
    celery_app.worker_main([
        "worker",
        f"--loglevel={settings.log_level.lower()}",
        "--concurrency=4",
        "--max-tasks-per-child=50",
        "-Q", "file_processing,default",
        "--without-gossip",
        "--without-mingle",
        "--without-heartbeat",
    ])

## 6. Updated Taskfile.yaml
version: '3'

vars:
  PYTHON: python
  PROJECT_NAME: genai_platform_services
  SOURCE_DIR: src
  TESTS_DIR: tests
  DIST_DIR: dist

tasks:
  # ... existing tasks ...

  run:
    desc: Run the FastAPI application
    cmds:
      - echo "Running FastAPI application..."
      - uv run uvicorn src.main:app --host=0.0.0.0 --port=8000 --reload

  run-celery:
    desc: Run Celery worker locally
    cmds:
      - echo "Running Celery worker..."
      - uv run python celery_worker.py

  run-both:
    desc: Run both FastAPI and Celery (requires supervisor)
    cmds:
      - echo "Starting both services..."
      - supervisord -c supervisor.conf

  celery-status:
    desc: Check Celery worker status
    cmds:
      - uv run celery -A src.celery_app inspect active

  celery-purge:
    desc: Purge all Celery tasks
    cmds:
      - uv run celery -A src.celery_app purge -f

## 7. Helm Deployment Options
Option A: Separate Deployments (Recommended)
Create two separate deployments - one for API, one for Celery workers.
helm-charts/dev-values/platform-service-api.yaml (rename existing):
# Keep existing config, add:
envVarsConfig:
  - name: SERVICE_MODE
    value: "api"
  # ... rest of existing env vars

helm-charts/dev-values/platform-service-celery.yaml (new file):
replicaCount: 2  # Scale workers independently

image:
  repository: "asia-south1-docker.pkg.dev/hbl-dev-gcp-gen-ai-prj-spk-5a/gen-ai-docker-repo/platform-service"
  pullPolicy: Always
  tag: M1.0.0

nameOverride: "platform-service-celery"

# Remove health checks for celery workers
healthcheck:
  livenessProbe:
    exec:
      command:
        - python
        - -c
        - "from src.celery_app import celery_app; celery_app.control.inspect().active()"
    periodSeconds: 60
    failureThreshold: 3
    initialDelaySeconds: 30

serviceAccount:
  create: false
  name: "ksa-genai"

# No service needed for workers
service:
  enabled: false

resources:
  limits:
    cpu: 2
    memory: 6Gi
  requests:
    cpu: 1
    memory: 4Gi

nodeSelector:
  node_pool: default-pool

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

envVarsConfig:
  - name: SERVICE_MODE
    value: "celery"
  - name: LOG_LEVEL
    value: "INFO"
  - name: REDIS_HOST
    value: "10.216.88.68"
  - name: REDIS_PORT
    value: "6378"
  - name: USE_SSL
    value: "True"
  - name: SSL_CA_CERTS
    value: "/tmp/redis/server-ca.pem"
  - name: MAX_CONCURRENT_FILES_PER_VS
    value: "5"
  - name: CELERY_TASK_TIME_LIMIT
    value: "1800"
  - name: MAX_TASK_RETRIES
    value: "1"
  # ... copy other necessary env vars from API config

extraVolumeMounts:
  - mountPath: /tmp/redis
    name: redis-ssl-secrets
    readOnly: true

extraVolumes:
  - name: redis-ssl-secrets
    secret:
      secretName: redis-ssl-secrets
      items:
        - key: server-ca.pem
          path: server-ca.pem

external_secrets:
  # Copy same secrets from API deployment
  - fileName: PLATFORM_SERVICE_DEV_DATABASE_URL.txt
    resourceName: projects/626624654237/secrets/PLATFORM_SERVICE_DEV_DATABASE_URL/versions/3
  # ... other needed secrets

Option B: Single Pod with Supervisor (Simpler but less scalable)
# Modify existing platform-service.yaml
envVarsConfig:
  - name: SERVICE_MODE
    value: "both"  # Runs both API and Celery
  # ... rest of config

## 8. pyproject.toml - Add Celery
[project]
dependencies = [
    # ... existing dependencies ...
    "celery[redis]>=5.3.4,<6.0.0",
]

## 9. Deployment Commands
# Build with Celery support
task docker-build

# Deploy API
helm upgrade --install platform-service-api ./helm-charts \
  -f helm-charts/dev-values/platform-service-api.yaml

# Deploy Celery workers (separate)
helm upgrade --install platform-service-celery ./helm-charts \
  -f helm-charts/dev-values/platform-service-celery.yaml

# OR deploy single pod with both
helm upgrade --install platform-service ./helm-charts \
  -f helm-charts/dev-values/platform-service.yaml

## 10. Local Development
# Terminal 1: Run FastAPI
task run

# Terminal 2: Run Celery
task run-celery

# Or run both with supervisor
task run-both

