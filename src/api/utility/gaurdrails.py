import requests
import json
import src.config as config
from src.logging_config import Logger

logger = Logger.create_logger(__name__)


def scan_prompt(prompt, session_id, usecase_id):
    headers = {'X-Session-ID': session_id,
               'X-Usecase-ID': usecase_id,
               'Content-Type': 'application/json'
               }
    data = {"guardrail_id": config.INPUT_PROMPT_GUARDRAIL_ID,
            "prompt": prompt}
    json_data = json.dumps(data) if data else None
    verify = False if config.IS_PROD.lower() == 'false' else True
    response = requests.post(config.GUARD_RAILS_INPUT_PROMPT_ENDPOINT,
                             headers=headers,
                             data=json_data, verify=verify)
    result = response.json()
    return result


def scan_output(input_prompt, output, session_id, usecase_id):
    headers = {'X-Session-ID': session_id,
               'X-Usecase-ID': usecase_id,
               'Content-Type': 'application/json'
               }
    data = {"guardrail_id": config.INPUT_PROMPT_GUARDRAIL_ID,
            "prompt": input_prompt,
            "output": output
            }
    json_data = json.dumps(data) if data else None
    response = requests.post(config.GUARD_RAILS_OUTPUT_PROMPT_ENDPOINT,
                             headers=headers,
                             data=json_data, verify=False)
    result = response.json()
    return result
