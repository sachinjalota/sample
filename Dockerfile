FROM asia-south1-docker.pkg.dev/hbl-dev-gcp-gen-ai-prj-spk-5a/genai-docker-virtual-repo/ubi9/python-311


USER root

ARG PYUSER

RUN echo $PYUSER

ARG TOKEN

WORKDIR /app

COPY requirements.txt .
RUN ls

# Install the application
RUN pip  install -r requirements.txt  --index-url https://$PYUSER:$TOKEN@asia-south1-python.pkg.dev/hbl-dev-gcp-gen-ai-prj-spk-5a/genai-python-virtual-repo/simple/


# Copy the rest of the application
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Switch back to non-root user
# USER 1001

# Expose the FastAPI port
EXPOSE 8080

# Start the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]