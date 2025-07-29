import weave
from typing import Optional, List, Dict, Any
import os
from src.ocr.extractor import extract_text_from_image_local
from src.rag.chunker import chunk_text_with_overlap
from src.rag.embed import create_embeddings
from src.rag.vectore_store import PineconeVectorStore
from src.rag.retriever import Retriever
from openai import OpenAI


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

    # LLM configuration
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.7

    def __init__(self, index_name: str, namespace: str, **kwargs):
        super().__init__(index_name=index_name, namespace=namespace, **kwargs)
        # We don't initialize the retriever here to avoid connecting to Pinecone
        # when the model is just instantiated. It will be lazy-loaded.
        self._retriever = None
        self._openai_client = None
    
    @property
    def openai_client(self):
        """Lazy-load the OpenAI client."""
        if self._openai_client is None:
            self._openai_client = OpenAI()
        return self._openai_client

    @property
    def retriever(self):
        """Lazy-load the retriever to avoid issues with Weave serialization."""
        if self._retriever is None:
            self._retriever = Retriever(index_name=self.index_name, namespace=self.namespace)
        return self._retriever

    @weave.op()
    def load_and_process_document(self, source_input: str, is_text_input: bool = False):
        """
        Processes a document from an image or direct text input.
        This represents the data ingestion pipeline.
        
        Args:
            source_input: Either an image path or direct text content
            is_text_input: If True, treat source_input as direct text; if False, as image path
        """
        # 1. Extract text (either from image or use direct input)
        if is_text_input:
            print("Using provided text input...")
            extracted_text = source_input
        else:
            print(f"Extracting text from {source_input}...")
            extracted_text = extract_text_from_image_local(source_input)
        
        # 2. Chunk the extracted text
        print("Chunking text...")
        text_chunks = self.chunk_document(extracted_text)
        
        # 3. Create embeddings and load into the vector store
        source_id = "text_input" if is_text_input else source_input
        return self.embed_and_load(text_chunks, source_id)

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
        This method performs the full RAG pipeline:
        1. Retrieves context from the vector store.
        2. Uses an LLM to generate an answer based on the context.
        """
        # 1. Retrieve context
        k = top_k if top_k is not None else self.retriever_top_k
        print(f"Retrieving context for query: '{query}' with top_k={k}")
        retrieved_matches = self.retriever.retrieve(query, top_k=k)
        
        context = [match.get('metadata', {}).get('text', '') for match in retrieved_matches]
        
        # 2. Generate an answer using the LLM
        print("Generating answer with LLM...")
        generated_answer = self.generate_answer(query, context)

        return {
            "query": query,
            "generated_answer": generated_answer,
            "retrieved_context": context,
            "raw_matches": retrieved_matches
        }

    def generate_answer(self, query: str, context: List[str]) -> str:
        """Generates an answer using the LLM based on the query and context."""
        context_str = "\n---\n".join(context)
        
        prompt = f"""
You are a friendly financial assistant. Use the provided document context to answer the user's question in a complete sentence.

=== Document Context ===
{context_str}
=== End Context ===

Question: {query}
Helpful Answer:
""".strip()
        
        response = self.openai_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.llm_temperature,
        )
        
        return response.choices[0].message.content.strip()

    @weave.op()
    def get_index_stats(self) -> Dict[str, Any]:
        """Returns statistics about the Pinecone index."""
        return self.retriever.vector_store.describe_index_stats() 