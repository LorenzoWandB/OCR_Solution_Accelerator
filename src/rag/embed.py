import os
from openai import OpenAI
from dotenv import load_dotenv
import weave
from typing import List, Union

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

@weave.op()
def create_embeddings(
    texts: Union[str, List[str]], 
    model: str = "text-embedding-3-small"
) -> List[List[float]]:
    """
    Creates embeddings for a list of texts using a specified OpenAI model.
    Always returns a list of embeddings.
    """
    client = OpenAI()
    
    # Ensure texts is a list
    if isinstance(texts, str):
        texts = [texts]
    
    # Replace newlines, which can cause issues with some models
    texts = [text.replace("\n", " ") for text in texts]
    
    if not texts:
        return []

    response = client.embeddings.create(input=texts, model=model)
    
    # Extract the embeddings from the response
    embeddings = [item.embedding for item in response.data]
    
    return embeddings
