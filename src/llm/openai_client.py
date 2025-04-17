import openai
import src.config as config
from src.models.generate_qna_payload import CompletionPayload
from src.utility.helper import parse_json_response
import httpx
from src.logging_config import Logger
from src.api.utility.gaurdrails import scan_prompt, scan_output
import traceback


class OpenAILLM:
    def __init__(self):
        self.logger = Logger.create_logger(__name__)
        self.client = self.return_openai_client()
        self.logger.info(f"Using LITELLM with {config.LITE_LLM_MODEL_ENDPOINT}")
        self.logger.debug(f"IS Prod : {config.IS_PROD}, {type(config.IS_PROD)}")

    def return_openai_client(self):
        verify = False if config.IS_PROD == 'False' else True
        http_client = httpx.Client(http2=True, verify=verify)
        return openai.OpenAI(api_key=config.LITE_LLM_API_KEY,
                             base_url=config.LITE_LLM_MODEL_ENDPOINT,
                             http_client=http_client)

    def _format_prompt(self, payload: CompletionPayload):
        try:
            placeholders = {
                "{no_of_qna}": str(payload.no_of_qna),
                "{question_context}": str(payload.question_context)
            }

            user_prompt = payload.user_prompt
            for placeholder, value in placeholders.items():
                user_prompt = user_prompt.replace(placeholder, value)

            final_prompt = f"{payload.system_prompt}\n\n{user_prompt}"

            return final_prompt
        except Exception as e:
            raise Exception(f"Prompt formatting error: {str(e)}")

    def _generate_llm_request_config(self, payload: CompletionPayload):
        return {
            "temperature": payload.temperature,
            "top_p": payload.top_p,
            "max_tokens": payload.max_completion_tokens,
        }

    def get_llm_response(self, prompt, config_params):
        self.logger.info(f"Type of prompt : ({type(prompt)})")
        messages = [
            {
                "role": "user",
                "content": prompt  # Just use the 'prompt' string directly
            }
        ]

        response = self.client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=messages,
            **config_params
        )
        final_response = response.choices[0].message.content
        return final_response

    def generate_response(self, payload: CompletionPayload, **kwargs):
        # self.logger.info(f"Input Payload: {payload.dict()}")
        try:
            prompt = self._format_prompt(payload)
            guard_rail_input_response = scan_prompt(prompt,
                                                    payload.session_id,
                                                    payload.usecase_id)
            self.logger.info(f"guard rail response: {guard_rail_input_response}")

            if not guard_rail_input_response['is_valid']:
                return {'error': 'Failed at Input Guardrail',
                        "scanners": guard_rail_input_response['scanners']}
            generate_content_config = self._generate_llm_request_config(payload)
            full_response = self.get_llm_response(prompt, generate_content_config)
            guard_rail_output_response = scan_output(prompt, full_response,
                                                     payload.session_id,
                                                     payload.usecase_id)
            if not guard_rail_output_response['is_valid']:
                return {'error': 'Failed at Output Guardrail',
                        "scanners": guard_rail_output_response['scanners']}
            parsed_response = parse_json_response(full_response)
            self.logger.info(f"Final Output Response (JSON): {parsed_response}")
            return parsed_response
        except Exception as e:
            self.logger.error(f"Failed in Openai request generation: {traceback.format_exc()}")
            self.logger.error(f"Error in content generation: {e}")
            return {"error": "Content generation error", "message": str(e)}
