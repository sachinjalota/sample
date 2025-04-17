# genai_platform_services
## Chat-As-Service
## How to Setup this service
- clone the Repo 
- setup your poetry 
    -  install poetry: ```pip install poetry```
    - install dependencies : ``` poetry install ```
    - activate the poetry shell : ``` poetry shell```
- Export Environment variables
```
export BASE_API_KEY="API-Key-configured-on-Proxy"
```
### While Testing in local ,export ssl certificate 
os.environ['SSL_CERT_FILE'] = "cacert.pem" 
This is required for fsspec to upload and download files

****
### Text Completion Endpoint

Endpoint Url: http://localhost:8080/UATEOL/API/text_completion <br>
#### text completion endpoint Sample request Payload
```json
{
  "model_name": "openai/gemini-1.5-flash",
  "system_prompt": "string",
  "user_prompt": "Who is the PM of Pakistan",
  "auth_params": {},
  "temperature": 1,
  "max_completion_tokens": -1,
  "max_tokens": -1,
  "modalities": [
    "text"
  ],
  "stream": false,
  "history": {
    "data": [
      {
        "role": "string",
        "content": "string"
      }
    ]
  }
}
```
#### Sample Response:
```json
{
  "id": "chatcmpl-441fb5be-d4b9-4280-9e9d-865de2eb8a09",
  "created": 1741692540,
  "model": "gemini-1.5-pro",
  "object": "chat.completion",
  "system_fingerprint": null,
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "\"string\" is a fundamental data type in computer programming. It represents a sequence of characters, like text. \n\nHere's a breakdown of what makes \"string\" important in programming:\n\n**What it is:**\n\n* **Sequence:**  Strings are ordered collections of characters, meaning the order of characters matters (\"hello\" is different from \"olleh\").\n* **Characters:**  These can include letters, numbers, spaces, punctuation marks, and even special symbols.\n\n**How it's used:**\n\n* **Storing Text:**  From names and addresses to entire articles, strings hold textual data.\n* **Displaying Information:** When you see words or sentences on your screen, a program likely used strings to present them.\n* **Manipulating Data:** Programming languages offer tools to work with strings, such as:\n    * **Concatenation:** Combining strings together (\"Hello\" + \" World\" = \"Hello World\")\n    * **Slicing:**  Extracting parts of a string (\"Example\"[0:4] gives you \"Exam\")\n    * **Searching:**  Finding specific characters or words within a string \n    * **Case Conversion:**  Changing between uppercase and lowercase (\"hello\".upper() becomes \"HELLO\")\n\n**Examples in different programming languages:**\n\n* **Python:** `\"Hello, world!\"`\n* **JavaScript:** `\"Hello, world!\"`\n* **Java:** `\"Hello, world!\"`\n* **C++:** `\"Hello, world!\"` \n\nLet me know if you'd like me to elaborate on any specific aspect of strings or provide more detailed examples in a particular programming language! \n",
        "role": "assistant",
        "tool_calls": null,
        "function_call": null
      }
    }
  ],
  "usage": {
    "completion_tokens": 342,
    "prompt_tokens": 2,
    "total_tokens": 344,
    "completion_tokens_details": null,
    "prompt_tokens_details": null
  },
  "service_tier": null,
  "vertex_ai_grounding_metadata": [],
  "vertex_ai_safety_results": [
    [
      {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "probability": "NEGLIGIBLE",
        "probabilityScore": 0.06347656,
        "severity": "HARM_SEVERITY_NEGLIGIBLE",
        "severityScore": 0.06933594
      },
      {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "probability": "NEGLIGIBLE",
        "probabilityScore": 0.04736328,
        "severity": "HARM_SEVERITY_NEGLIGIBLE",
        "severityScore": 0.05419922
      },
      {
        "category": "HARM_CATEGORY_HARASSMENT",
        "probability": "NEGLIGIBLE",
        "probabilityScore": 0.091308594,
        "severity": "HARM_SEVERITY_NEGLIGIBLE",
        "severityScore": 0.040283203
      },
      {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "probability": "NEGLIGIBLE",
        "probabilityScore": 0.045410156,
        "severity": "HARM_SEVERITY_NEGLIGIBLE",
        "severityScore": 0.024414063
      }
    ]
  ],
  "vertex_ai_citation_metadata": []
}
```
- Reference : https://platform.openai.com/docs/api-reference/chat/create
****
### Image Completion Endpoint 
Endpoint Url: http://localhost:8080/UATEOL/API/image_completion <br>
#### image completion endpoint Sample request payload
```json
{
  "model_name": "openai/gemini-1.5-flash",
  "system_prompt": "string",
  "user_query": "What is in the Image",
  "image_url": "https://general-purpose-public.s3.ap-south-1.amazonaws.com/gemini_test_images/aadharCard.jpeg",
  "auth_params": {},
  "max_completion_tokens": -1,
  "max_tokens": -1,
  "modalities": [
    "text"
  ]
}
```
#### Sample Image completion response :
```json
{
  "id": "chatcmpl-c1a58c60-a287-4135-8c36-41e10ec6f4d7",
  "created": 1741693222,
  "model": "gemini-1.5-pro",
  "object": "chat.completion",
  "system_fingerprint": null,
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "The image is a card about an episode of the Indian television show \"Janhit Mein Jaari.\"\n\nHere's what the card tells us:\n\n* **Government of India:** The card is issued or endorsed by the Indian government.\n* **Name: Kettan Singh:**  This likely refers to the host, a guest speaker, or someone prominently featured in this episode.\n* **Show: Janhit Mein Jaari:** This is the name of the TV show.\n* **Topic: Aadhaaritis:** This episode focuses on Aadhaar, likely discussing its benefits, issues, or related policies. \n* **4444 5555 6666:** This could be a helpline number, a code related to the episode, or an identifier for Aadhaar-related services.\n* **आधार - आम आदमी का अधिकार  (Aadhaar - Common Man's Right):** This tagline emphasizes that Aadhaar is a right for ordinary citizens.\n\nThe QR code likely leads to more information about the show, the episode, or Aadhaar itself. \n",
        "role": "assistant",
        "tool_calls": null,
        "function_call": null
      }
    }
  ],
  "usage": {
    "completion_tokens": 228,
    "prompt_tokens": 264,
    "total_tokens": 492,
    "completion_tokens_details": null,
    "prompt_tokens_details": null
  },
  "service_tier": null,
  "vertex_ai_grounding_metadata": [],
  "vertex_ai_safety_results": [
    [
      {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "probability": "NEGLIGIBLE",
        "probabilityScore": 0.09814453,
        "severity": "HARM_SEVERITY_NEGLIGIBLE",
        "severityScore": 0.09423828
      },
      {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "probability": "NEGLIGIBLE",
        "probabilityScore": 0.16210938,
        "severity": "HARM_SEVERITY_NEGLIGIBLE",
        "severityScore": 0.06298828
      },
      {
        "category": "HARM_CATEGORY_HARASSMENT",
        "probability": "NEGLIGIBLE",
        "probabilityScore": 0.091308594,
        "severity": "HARM_SEVERITY_NEGLIGIBLE",
        "severityScore": 0.06298828
      },
      {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "probability": "NEGLIGIBLE",
        "probabilityScore": 0.08886719,
        "severity": "HARM_SEVERITY_NEGLIGIBLE",
        "severityScore": 0.021606445
      }
    ]
  ],
  "vertex_ai_citation_metadata": []
}
```
- References: https://platform.openai.com/docs/guides/images
