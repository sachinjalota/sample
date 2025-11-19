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
