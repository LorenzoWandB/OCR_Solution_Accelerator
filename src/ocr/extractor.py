import os
import weave
import base64

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class OCRExtractor:
    def __init__(self):
        self.client = OpenAI()
        self.default_prompt = """
From the document image, extract all text content verbatim and return it as a single block of text.
Preserve the original structure, including lines and spacing, as much as possible.
Do not summarize, interpret, or format the text as JSON.
"""
    
    def extract_text_from_image_local(self, image_path: str, prompt: str = None) -> str:
        return extract_text_from_image_local(image_path, prompt or self.default_prompt)


@weave.op()
def extract_text_from_image_local(image_path: str, prompt: str = None) -> str:
    if prompt is None:
        prompt = """
From the document image, extract all text content verbatim and return it as a single block of text.
Preserve the original structure, including lines and spacing, as much as possible.
Do not summarize, interpret, or format the text as JSON.
"""
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    
    extracted_text = extract_text_from_image(image_base64, prompt)

    # Create a new file path for the extraction
    file_name_without_ext, _ = os.path.splitext(image_path)
    output_path = f"{file_name_without_ext}.txt"

    with open(output_path, "w") as f:
        f.write(extracted_text)

    return extracted_text
    
@weave.op()
def extract_text_from_image(image_base64: str, prompt: str) -> str:
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[

            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    
    return response.choices[0].message.content