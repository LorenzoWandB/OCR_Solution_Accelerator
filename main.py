import weave
from dotenv import load_dotenv
import os
from src.weave.model import RagModel

load_dotenv()

# --- Configuration ---
PROJECT = "Solution-Accelerator-MRM"
PINECONE_INDEX_NAME = "mrm-rag-weave" # Using a new index for clarity
PINECONE_NAMESPACE = "financial_statements"
IMAGE_PATH = "data/images/simple_statement.jpg"

def main():
    # Initialize Weave
    weave.init(PROJECT)

    # --- Model Initialization ---
    # Instantiate the RagModel, which encapsulates the entire RAG pipeline.
    # Weave will track the configuration of this model.
    print("Initializing RAG Model...")
    rag_model = RagModel(
        index_name=PINECONE_INDEX_NAME,
        namespace=PINECONE_NAMESPACE,
        embedding_model="text-embedding-3-small",
        chunk_size=500,
        chunk_overlap=100,
        retriever_top_k=5
    )
    # We can publish the model to Weave to version and share it.
    # Any change to the model's configuration will create a new version.
    model_ref = weave.publish(rag_model, "my-rag-model")
    print(f"Model published to Weave: {model_ref}")


    # --- Data Ingestion ---
    # Process and load a document into the vector store.
    # This is a one-time operation per document.
    print("\n--- Ingesting Document ---")
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file not found at {IMAGE_PATH}")
    else:
        # The load_and_process_document method is a weave.op, so all its
        # substeps (extraction, chunking, embedding) will be traced.
        ingestion_result = rag_model.load_and_process_document(IMAGE_PATH)
        print(f"Ingestion complete. Processed {ingestion_result['processed_chunks']} chunks.")
    
    # --- Querying ---
    # Use the model's predict method to retrieve context for a query.
    print("\n--- Querying the Model ---")
    
    query = "What is the Net Profit?"
    print(f"Query: '{query}'")

    # The predict method is also a weave.op. Calling it creates a Weave call record
    # that links the specific model version, inputs, and outputs.
    prediction = rag_model.predict(query)

    print("\n--- Prediction Results ---")
    print(f"Retrieved context for query: '{prediction['query']}'")
    
    if prediction['retrieved_context']:
        print("Retrieved Context:")
        for i, context_chunk in enumerate(prediction['retrieved_context']):
            print(f"  [{i+1}] \"{context_chunk.strip()}\"")
    else:
        print("No context was retrieved.")

if __name__ == "__main__":
    main()



