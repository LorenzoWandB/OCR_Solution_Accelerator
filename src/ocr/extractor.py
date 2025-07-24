import os
import weave
import base64

from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional
import mimetypes

load_dotenv()

client = OpenAI()

@weave.op()
def extract_text_from_image_local(image_path: str) -> str:
    # Guess the MIME type of the image
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for {image_path}")

    # Read the image file in binary mode and encode it in base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text from this image, including any numbers and symbols. Provide only the raw text content without any additional formatting or commentary."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=2000,
    )
    
    return response.choices[0].message.content or ""