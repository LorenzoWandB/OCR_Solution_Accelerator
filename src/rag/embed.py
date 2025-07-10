import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def create_embeddings(text, model="text-embedding-3-small"):
    if not text:
        print("Error: Input text is empty.")
        return None

    try:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"An error occurred while creating embeddings: {e}")
        return None

def embed_from_file(file_path, model="text-embedding-3-small"):
    text = read_text_file(file_path)
    if text is None:
        return None
    
    return create_embeddings(text, model)
