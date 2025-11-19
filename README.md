>> /genai_platform_services/Dockerfile
FROM asia-south1-docker.pkg.dev/hbl-dev-gcp-gen-ai-prj-spk-5a/genai-docker-virtual-repo/ubi9/python-311

USER root

WORKDIR /app

# Pass private PyPI credentials as build arguments
ARG GCP_ARTIFACT_USERNAME=oauth2accesstoken
ARG GCP_ARTIFACT_PASSWORD
ARG PYPI_REPO_URL=asia-south1-python.pkg.dev/hbl-dev-gcp-gen-ai-prj-spk-5a/genai-python-virtual-repo/simple/

# Set environment variables
ENV APP_ROOT=/app/.venv \
    VIRTUAL_ENV=${APP_ROOT} \
    UV_PROJECT_ENVIRONMENT=${APP_ROOT} \
    UV_DEFAULT_INDEX=https://$GCP_ARTIFACT_USERNAME:$GCP_ARTIFACT_PASSWORD@$PYPI_REPO_URL \
    PATH="${APP_ROOT}/bin:$PATH" \
    APP_HOST="0.0.0.0" \
    APP_PORT=8080

RUN pip install --no-cache-dir --index-url https://$GCP_ARTIFACT_USERNAME:$GCP_ARTIFACT_PASSWORD@$PYPI_REPO_URL uv==0.5.28
RUN pip install --no-cache-dir --index-url https://$GCP_ARTIFACT_USERNAME:$GCP_ARTIFACT_PASSWORD@$PYPI_REPO_URL --upgrade pip setuptools wheel

COPY pyproject.toml .
COPY uv.lock .
COPY README.md .

# Install Dependencies
RUN uv sync --no-dev

COPY src ./src

COPY init.sh .
RUN chmod +x init.sh

# Expose the FastAPI port
EXPOSE ${APP_PORT}

RUN chmod -R +x /app

RUN chown -R 1001:0 /app

USER 1001

# Start the application
CMD ["./init.sh"]


>> /genai_platform_services/Taskfile.yaml
version: '3'

vars:
  PYTHON: python
  PROJECT_NAME: genai_platform_services
  SOURCE_DIR: src
  TESTS_DIR: tests
  DIST_DIR: dist

tasks:
  lint:
    desc: Run ruff for lint
    cmds:
      - echo "Running ruff for lint..."
      - 'uv run ruff check {{.SOURCE_DIR}} {{.TESTS_DIR}}'

  format:
    desc: Run isort and ruff format for code formatting
    cmds:
      - echo "Running isort and ruff format for code formatting..."
      - 'uv run isort {{.SOURCE_DIR}} {{.TESTS_DIR}}'
      - 'uv run ruff format {{.SOURCE_DIR}} {{.TESTS_DIR}}'

  type-check:
    desc: Runs mypy type check on the code
    cmds:
      - echo "Running mypy check on the code"
      - uv run mypy .

  test:
    desc: Run tests with pytest
    cmds:
      - echo "Running tests with pytest..."
      - 'ENV_FILE=".env.test" PYTHONWARNINGS="ignore::DeprecationWarning" pytest --cov={{.SOURCE_DIR}} --cov-report=html --cov-report=term-missing'

  build:
    desc: Build the project with poetry
    cmds:
      - echo "Building the project with poetry..."
      - 'uv build'

  docker-build:
    desc: Command to build the docker image
    cmds:
      - echo "Building docker image"
      - docker build -t asia-south1-docker.pkg.dev/hbl-dev-gcp-gen-ai-prj-spk-5a/gen-ai-docker-repo/genai_platform_services --build-arg GCP_ARTIFACT_USERNAME="oauth2accesstoken" --build-arg GCP_ARTIFACT_PASSWORD=$(gcloud auth print-access-token) -v /mnt/filestore:/mnt/filestore   .

  clean:
    desc: Clean up build and distribution directories
    cmds:
      - echo "Cleaning up build and distribution directories..."
      - rm -rf {{.DIST_DIR}} *.egg-info

  setup:
    desc: Set up the virtual environment for project
    cmds:
      - echo "Setting up the virtual env for project..."
      - python -m venv .venv
      - source .venv/bin/activate
      - uv sync
      - pre-commit install

  run:
    desc: Run the application
    cmds:
      - echo "Running the application..."
      - uv run uvicorn src.main:app --host=0.0.0.0 --port=8000 --reload

#  generate-requirements:
#    desc: Export requirements.txt from poetry
#    cmds:
#      - 'uv export -f requirements.txt --without-urls --without-hashes --only main | cut -d ";" -f1 > requirements.txt'




>> helm-charts/dev-values/platform-service.yaml
replicaCount: 1

image:
  repository: "asia-south1-docker.pkg.dev/hbl-dev-gcp-gen-ai-prj-spk-5a/gen-ai-docker-repo/platform-service"
  pullPolicy: Always
  tag: M1.0.0


nameOverride: "platform-service"
fullnameOverride: ""

healthcheck:
  livenessProbe:
    httpGet:
      path: /DEV/platform-service/v1/api/health
      port: http
    periodSeconds: 30
    failureThreshold: 4
    initialDelaySeconds: 10
    timeoutSeconds: 60
  readinessProbe:
    httpGet:
      path: /DEV/platform-service/v1/api/health
      port: http
    periodSeconds: 10
    initialDelaySeconds: 10
    timeoutSeconds: 60

timeoutseconds: 300

serviceAccount:
  # Specifies whether a service account should be created
  create: false
  # Annotations to add to the service account
  annotations: { }
  # The name of the service account to use.
  # If not set and create is true, a name is generated using the fullname template
  name: "ksa-genai"

podAnnotations: { }

podSecurityContext:
  { }
# fsGroup: 2000

securityContext:
  { }
  # readOnlyRootFilesystem: true
  # runAsNonRoot: true
# allowPrivilegeEscalation: false

service:
  enabled: true
  type: ClusterIP
  port: 80
  containerPort: 8080

resources:
  # We usually recommend not to specify default resources and to leave this as a conscious
  # choice for the user. This also increases chances charts run on environments with little
  # resources, such as Minikube. If you do want to specify resources, uncomment the following
  # lines, adjust them as necessary, and remove the curly braces after 'resources:'.
  limits:
    cpu: 2
    memory: 4Gi
  requests:
    cpu: 1
    memory: 4Gi

nodeSelector:
  node_pool: default-pool

autoscaling:
  enabled: false
  minReplicas: 1
  maxReplicas: 1
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 70


envVarsConfig:
  - name: IS_PROD
    value: "False"
  - name: BASE_URL
    value: "http://localhost:8000"
  - name: CLOUD_STORAGE_PROVIDER
    value:  "gcp"
  - name: LOG_LEVEL
    value: "DEBUG"
  - name: LOG_PATH
    value: "/app/logs/app.log"
  - name: CLOUD_PROVIDER
    value: "gcp"
  - name: UPLOAD_BUCKET_NAME
    value: "genai-ai-utilities-storage"
  - name: UPLOAD_FOLDER_NAME
    value: "uploads"
  - name: UPLOAD_FILE_LIMIT
    value: "10485760"
  - name: LLM_PROVIDER
    value: "gemini"
  - name: HEALTH_CHECK_ENDPOINT
    value: "health"
  - name: UPLOAD_OBJECT_ENDPOINT
    value: "/upload_object"
  - name: GENERATE_QNA_ENDPOINT
    value: "/generate_qna"
  - name: GCP_LOCATION
    value: "asia-south1"
  - name: GCP_PROJECT
    value: "hbl-dev-gcp-gen-ai-prj-spk-5a"
  - name: GEMINI_MODEL
    value: "google/gemini-1.5-flash-002"
  - name: GCP_API_BASE_URL
    value: "'https://${GCP_LOCATION}-aiplatform.googleapis.com/v1beta1'"
  - name: GCP_AUTH_SCOPE
    value: "https://www.googleapis.com/auth/cloud-platform"
  - name: APP_WORKERS
    value: "-1"
  - name: GOOGLE_CLOUD_PROJECT
    value: "hbl-dev-gcp-gen-ai-prj-spk-5a"
  - name: SERVICE_SLUG
    value: "platform-service"
  - name: API_COMMON_PREFIX
    value: "/v1/api"
  - name: SSL_CA_CERTS
    value: "/tmp/redis/server-ca.pem"
  - name: REDIS_HOST
    value: "10.216.88.68"
  - name: REDIS_PORT
    value: "6378"
  - name: USE_SSL
    value: "True"
  - name: BASE_API_URL
    value: "http://litellm/DEV/litellm"
  - name: GUARDRAILS_ENDPOINT
    value: "http://genai-guardrail-service/DEV/guardrails/"
  - name: PROMPT_HUB_ENDPOINT
    value: "http://prompt-hub-service/DEV/prompthub-service"
  - name: OPIK_WORKSPACE
    value: "r36386"
  - name: OPIK_PROJECT_NAME 
    value: "default"
  - name: OPIK_CHECK_TLS_CERTIFICATE
    value: "false"
  - name: OPIK_URL_OVERRIDE
    value: "https://10.216.70.62/opik/api"
  - name: ALLOW_HEADERS
    value: "Authorization,Content-Type,Accept,Origin,User-Agent,X-Requested-With,X-API-Key,X-Session-Id,X-Usecase-Id,X-Correlation-ID,x-base-api-key,token"
  - name: ALLOWED_ORIGINS
    value: "https://10.216.70.62,http://localhost:3000"
  - name: DEFAULT_MODEL
    value: "gemini-2.5-flash"
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
  - fileName: PLATFORM_SERVICE_DEV_LITE_LLM_API_KEY.txt
    resourceName: projects/626624654237/secrets/PLATFORM_SERVICE_DEV_LITE_LLM_API_KEY/versions/3
  - fileName: PLATFORM_SERVICE_DEV_BASE_API_KEY.txt
    resourceName: projects/626624654237/secrets/PLATFORM_SERVICE_DEV_BASE_API_KEY/versions/1
  - fileName: PLATFORM_SERVICE_DEV_DATABASE_URL.txt
    resourceName: projects/626624654237/secrets/PLATFORM_SERVICE_DEV_DATABASE_URL/versions/3
  - fileName: PLATFORM_SERVICE_DEV_PLAYGROUND_API_KEY.txt
    resourceName: projects/626624654237/secrets/PLATFORM_SERVICE_DEV_PLAYGROUND_API_KEY/versions/1
  - fileName: PLATFORM_SERVICE_DEV_MASTER_API_KEY.txt
    resourceName: projects/626624654237/secrets/PLATFORM_SERVICE_DEV_MASTER_API_KEY/versions/1
  - fileName: PLATFORM_SERVICE_DEV_DEFAULT_API_KEY.txt
    resourceName: projects/626624654237/secrets/PLATFORM_SERVICE_DEV_DEFAULT_API_KEY/versions/1
  - fileName: PLATFORM_SERVICE_DEV_OPIK_API_KEY.txt
    resourceName: projects/626624654237/secrets/PLATFORM_SERVICE_DEV_OPIK_API_KEY/versions/1
