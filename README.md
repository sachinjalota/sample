[hdfcbank@genai-hdfc-jump-vm genai-agent-mesh]$ sudo docker run asia-south1-docker.pkg.dev/hbl-dev-gcp-gen-ai-prj-spk-5a/gen-ai-docker-repo/litellm-pg2bq-sync:v20251215
Emulate Docker CLI using podman. Create /etc/containers/nodocker to quiet msg.
<frozen runpy>:128: RuntimeWarning: 'litellm_sync.sync' found in sys.modules after import of package 'litellm_sync', but prior to execution of 'litellm_sync.sync'; this may result in unpredictable behaviour
2025-12-15 15:35:54,673 - __main__ - ERROR - Fatal error: Config file not found: config/config.json
Traceback (most recent call last):
  File "/app/src/litellm_sync/sync.py", line 445, in main
    sync = LiteLLMBigQuerySync(config_path="config/config.json")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/src/litellm_sync/sync.py", line 81, in __init__
    self.config = self._load_config(config_path)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/src/litellm_sync/sync.py", line 92, in _load_config
    raise FileNotFoundError(f"Config file not found: {config_path}")
FileNotFoundError: Config file not found: config/config.json
