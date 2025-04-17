import json
import os
import pymupdf4llm


def parse_json_response(response: str):
    cleaned_response = response.strip("```json").strip()
    cleaned_response = cleaned_response.strip("```").strip()

    try:
        parsed_response = json.loads(cleaned_response)
        return parsed_response
    except json.JSONDecodeError as e:
        return {"error": "Invalid JSON output", "message": str(e)}


def get_markdown_from_pdf(file_path: str) -> str:
    if os.path.exists(file_path) and file_path.endswith('.pdf'):
        markdown_string = pymupdf4llm.to_markdown(file_path)
        return markdown_string
    else:
        return  ''
