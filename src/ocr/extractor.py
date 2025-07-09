import os
import json
import weave
import base64

from dotenv import load_dotenv
from openai import OpenAI
from src.weave.prompt import extractor_prompt

load_dotenv()


def extract_text_from_image_local(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    
    extracted_text = extract_text_from_image(image_base64)

    # Create a new file path for the extraction
    file_name_without_ext, _ = os.path.splitext(image_path)
    output_path = f"{file_name_without_ext}.txt"

    with open(output_path, "w") as f:
        f.write(extracted_text)

    return extracted_text
    
@weave.op()
def extract_text_from_image(image_base64: str) -> str:
    client = OpenAI()
    
    prompt = extractor_prompt
    
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
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