podman build -t litellm-pg2bq-sync:latest .
podman tag litellm-pg2bq-sync:latest asia-south1-docker.pkg.dev/hbl-dev-gcp-gen-ai-prj-spk-5a/genai-docker-virtual-repo/litellm-pg2bq-sync:latest
podman push asia-south1-docker.pkg.dev/hbl-dev-gcp-gen-ai-prj-spk-5a/genai-docker-virtual-repo/litellm-pg2bq-sync:latest
kubectl apply -f k8s-cronjob.yaml
