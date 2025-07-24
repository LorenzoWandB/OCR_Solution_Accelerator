import weave
from typing import Optional, List, Dict, Any
import os
from src.ocr.extractor import extract_text_from_image_local
from src.rag.chunker import chunk_text_with_overlap
from src.rag.embed import create_embeddings
from src.rag.vectore_store import PineconeVectorStore
from src.rag.retriever import Retriever

class RagModel(weave.Model):
    # --- Configuration Attributes ---
    # These attributes define the configuration of our RAG pipeline.
    # Weave tracks these, and any change creates a new version of the model.
    
    # Pinecone configuration
    index_name: str
    namespace: str

    # Embedding model
    embedding_model: str = "text-embedding-3-small"
    
    # Chunking configuration
    chunk_size: int = 500
    chunk_overlap: int = 100

    # Retriever configuration
    retriever_top_k: int = 5

    def __init__(self, index_name: str, namespace: str, **kwargs):
        super().__init__(index_name=index_name, namespace=namespace, **kwargs)
        # We don't initialize the retriever here to avoid connecting to Pinecone
        # when the model is just instantiated. It will be lazy-loaded.
        self._retriever = None
    
    @property
    def retriever(self):
        """Lazy-load the retriever to avoid issues with Weave serialization."""
        if self._retriever is None:
            self._retriever = Retriever(index_name=self.index_name, namespace=self.namespace)
        return self._retriever

    @weave.op()
    def load_and_process_document(self, image_path: str):
        """
        Processes a document from an image, from extraction to vector store upload.
        This represents the data ingestion pipeline.
        """
        # 1. Extract text from the image
        print(f"Extracting text from {image_path}...")
        extracted_text = extract_text_from_image_local(image_path)
        
        # 2. Chunk the extracted text
        print("Chunking text...")
        text_chunks = self.chunk_document(extracted_text)
        
        # 3. Create embeddings and load into the vector store
        return self.embed_and_load(text_chunks, image_path)

    @weave.op()
    def chunk_document(self, text: str) -> List[Dict[str, Any]]:
        """Chunks the given text using the model's configuration."""
        return chunk_text_with_overlap(
            text, 
            chunk_size=self.chunk_size, 
            overlap=self.chunk_overlap
        )

    @weave.op()
    def embed_and_load(self, chunks: List[Dict[str, Any]], source_id: str) -> Dict[str, Any]:
        """Embeds text chunks and loads them into the vector store."""
        # Create embeddings for each chunk
        print(f"Creating embeddings for {len(chunks)} chunks...")
        texts_to_embed = [chunk['text'] for chunk in chunks]
        embeddings = create_embeddings(texts_to_embed, model=self.embedding_model)

        # Prepare vectors for Pinecone
        vectors_to_upsert = []
        doc_basename = os.path.basename(source_id)
        for i, chunk in enumerate(chunks):
            vectors_to_upsert.append({
                "id": f"doc_{doc_basename}_{i}",
                "values": embeddings[i],
                "metadata": {
                    "text": chunk['text'],
                    "filename": doc_basename,
                    "chunk_number": i
                }
            })

        # Upsert vectors into Pinecone
        print(f"Upserting {len(vectors_to_upsert)} vectors into Pinecone...")
        self.retriever.vector_store.upsert(vectors_to_upsert)
        
        print("Document processing and loading complete.")
        return {"processed_chunks": len(chunks)}

    @weave.op()
    def predict(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        This method performs the retrieval part of the RAG pipeline.
        It takes a query and returns the retrieved context.
        """
        k = top_k if top_k is not None else self.retriever_top_k
        print(f"Retrieving context for query: '{query}' with top_k={k}")
        retrieved_matches = self.retriever.retrieve(query, top_k=k)
        
        # For now, we'll just return the retrieved context.
        # In a full implementation, you would feed this to an LLM.
        context = [match.get('metadata', {}).get('text', '') for match in retrieved_matches]

        return {
            "query": query,
            "retrieved_context": context,
            "raw_matches": retrieved_matches
        }

    @weave.op()
    def get_index_stats(self) -> Dict[str, Any]:
        """Returns statistics about the Pinecone index."""
        return self.retriever.vector_store.describe_index_stats() 