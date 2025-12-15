FROM asia-south1-docker.pkg.dev/hbl-dev-gcp-gen-ai-prj-spk-5a/genai-docker-virtual-repo/ubi9/python-311

USER root

WORKDIR /app

ARG GCP_ARTIFACT_USERNAME=oauth2accesstoken
ARG GCP_ARTIFACT_PASSWORD
ARG PYPI_REPO_URL=asia-south1-python.pkg.dev/hbl-dev-gcp-gen-ai-prj-spk-5a/genai-python-virtual-repo/simple/
ARG PYPI_LOCAL_REPO_URL=asia-south1-python.pkg.dev/hbl-dev-gcp-gen-ai-prj-spk-5a/genai-python-repo/simple/

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY config/ ./config/

# Install Python dependencies
RUN pip install --no-cache-dir --index-url https://$GCP_ARTIFACT_USERNAME:$GCP_ARTIFACT_PASSWORD@$PYPI_REPO_URL --upgrade pip && \
    pip install --no-cache-dir --index-url https://$GCP_ARTIFACT_USERNAME:$GCP_ARTIFACT_PASSWORD@$PYPI_REPO_URL -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Kolkata

USER 1001

# Run the sync script
CMD ["python", "-m", "litellm_sync.sync"]
