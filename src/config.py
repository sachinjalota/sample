import os
from dotenv import load_dotenv

try:
    load_dotenv(".env")
except Exception as e:
    print("Unable to load dotenv ")
    pass
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
ALLOW_CREDENTIALS = os.getenv("ALLOW_CREDENTIALS", "true").lower() == "true"
ALLOW_METHODS = os.getenv("ALLOW_METHODS", "GET,POST,OPTIONS")
ALLOW_HEADERS = os.getenv("ALLOW_HEADERS", "*")

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
LOG_PATH = os.getenv("LOG_PATH", "../logs/app.log")

SERVICE_SLUG = os.getenv("SERVICE_SLUG", 'platform-service')
ENV = os.getenv("ENV", "DEV")

TEXT_COMPLETION_ENDPOINT = os.getenv("TEXT_COMPLETION_ENDPOINT", "/text_completion")
IMAGE_COMPLETION_ENDPOINT = os.getenv("TEXT_COMPLETION_ENDPOINT", "/image_completion")

# CLOUD UPLOADS
CLOUD_STORAGE_PROVIDER = os.getenv("CLOUD_STORAGE_PROVIDER", "gcp")  # Can take  'aws'
UPLOAD_FILE_LIMIT = int(os.getenv("UPLOAD_FILE_LIMIT", 10485760))  # 10 MB
UPLOAD_BUCKET_NAME = os.getenv("UPLOAD_BUCKET_NAME", "genai-ai-utilities-storage")
UPLOAD_FOLDER_NAME = os.getenv("UPLOAD_FOLDER_NAME", "uploads")
UPLOAD_OBJECT_ENDPOINT = os.getenv("UPLOAD_OBJECT_ENDPOINT", 'upload')
#

API_COMMON_PREFIX = os.getenv("API_COMMON_PREFIX", "/v1/api")
HEALTH_CHECK = os.getenv("HEALTH_CHECK", "/health")
GENERATE_QNA_ENDPOINT = os.getenv("GENERATE_QNA_ENDPOINT", "generate_qna")

# DEFAULT_TEXT_COMPLETION_MODEL = os.getenv("DEFAULT_TEXT_COMPLETION_MODEL", "vertex_ai/gemini-1.5-flash-002")
DEFAULT_TEXT_COMPLETION_MODEL = os.getenv("DEFAULT_TEXT_COMPLETION_MODEL", "openai/gemini-1.5-flash")
DEFAULT_IMAGE_COMPLETION_MODEL = os.getenv("DEFAULT_IMAGE_COMPLETION_MODEL", "openai/gemini-1.5-flash")

BASE_API_URL = os.getenv("BASE_API_URL", "http://gemini.quickpocdemo.com")

BASE_API_KEY = os.getenv("BASE_API_KEY", '')
BASE_API_HEADERS = {'x-goog-api-key': os.getenv("BASE_API_KEY", '')}
AVAILABLE_MODELS = os.getenv("AVAILABLE_MODELS", "vertex_ai/gemini-1.5-flash-002,")

LITE_LLM_API_KEY = os.getenv("LITE_LLM_API_KEY", "")
LLM_MODEL_TYPE = os.getenv("LLM_MODEL_TYPE", 'litellm')  # Can take gemini
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", 'gemini-1.5-flash')  # can take
LITE_LLM_MODEL_ENDPOINT = os.getenv("LITE_LLM_MODEL_ENDPOINT", "https://10.216.70.62/DEV/litellm")

# Guard Rails
GUARD_RAILS_INPUT_PROMPT_ENDPOINT = os.getenv("GUARD_RAILS_INPUT_PROMPT_ENDPOINT",
                                              "https://10.216.70.62/DEV/guardrails/api/v1/analyze/prompt")

INPUT_PROMPT_GUARDRAIL_ID = os.getenv("INPUT_PROMPT_GUARDRAIL_ID", 1)

GUARD_RAILS_OUTPUT_PROMPT_ENDPOINT = os.getenv("GUARD_RAILS_INPUT_PROMPT_ENDPOINT",
                                               "https://10.216.70.62/DEV/guardrails/api/v1/analyze/output")

OUTPUT_GUARDRAIL_ID = os.getenv("OUTPUT_GUARDRAIL_ID", 2)

IS_PROD = os.getenv("IS_PROD", "True")
