import weave
from src.ocr.extractor import extract_text_from_image_local
from src.rag.embed import create_embeddings
from src.rag.vectore_store import PineconeVectorStore
from dotenv import load_dotenv
import os

from src.rag.retriever import Retriever

load_dotenv()

PROJECT = "Solution-Accelerator-MRM"
PINECONE_INDEX_NAME = "mrm-rag"
PINECONE_NAMESPACE = ""
IMAGE_PATH = "data/images/simple_statement.jpg"

weave.init(PROJECT)

extractor_prompt = """
Extract all readable text from this image. Format the extracted entities as a valid JSON.
Do not return any extra text, just the JSON. Do not include ```json```
"""
system_prompt = weave.StringPrompt(extractor_prompt)
weave.publish(system_prompt, name="Extractor-Prompt")


@weave.op()
def process_image(image_path: str):
    # 1. Extract text from image
    print(f"Extracting text from {image_path}...")
    extracted_text = extract_text_from_image_local(image_path, extractor_prompt)
    print("Text extracted successfully.")

    # 2. Create embeddings
    print("Creating embeddings...")
    embedding = create_embeddings(extracted_text)
    if embedding is None:
        print("Failed to create embeddings.")
        return
    print("Embeddings created successfully.")

    # 3. Store in Pinecone
    print("Initializing Pinecone vector store...")
    vector_store = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        namespace=PINECONE_NAMESPACE
    )
    # The dimension for text-embedding-3-small is 1536
    vector_store.initialize_index(dimension=1536)

    image_id = os.path.basename(image_path)

    vectors_to_upsert = [{
        "id": image_id,
        "values": embedding,
        "metadata": {"text": extracted_text}
    }]

    print(f"Upserting vector for {image_id} into Pinecone.")
    vector_store.upsert(vectors=vectors_to_upsert)
    print("Vector upserted successfully.")


if __name__ == "__main__":
    # By default, this script will now run the retrieval process.
    # To ingest a new image, uncomment the lines below.
    # -----------------------------------------------------------
    print("--- Ingesting Image ---")
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file not found at {IMAGE_PATH}")
        print("Please update the IMAGE_PATH variable in main.py with the correct path to your image.")
    else:
        process_image(IMAGE_PATH)
    print("\n--- Ingestion Complete ---\n")
    # -----------------------------------------------------------

    # --- Querying Index ---
    print("--- Running Retrieval and Reranking ---")
    retriever = Retriever(
        index_name=PINECONE_INDEX_NAME,
        namespace=PINECONE_NAMESPACE
    )
    
    # Example query - change this to ask a question about your document
    query = "What is the Net Profit?"
    
    print(f"Query: '{query}'")
    results = retriever.retrieve_and_rerank(query)

    if results:
        print("\nTop search results:")
        for i, match in enumerate(results):
            score = match.get('score', 0)
            text = match.get('document', {}).get('text', 'No text found')
            print(f"  {i+1}. Score: {score:.4f}")
            print(f"     Text: \"{text.strip()}\"\n")
    else:
        print("No results found.")



