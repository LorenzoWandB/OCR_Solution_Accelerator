import weave
from typing import Optional, List, Dict, Any
import os
import asyncio
from src.llamaindex.extractor import extract_documents
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
        # We don't initialize components here to avoid connecting to external services
        # when the model is just instantiated. They will be lazy-loaded.
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
    async def load_and_process_document(
        self, 
        source_input: Any, # Can now be a file path or a file-like object
        document_type: Optional[str] = None # document_type is now handled by the schema in the extractor
    ):
        """
        Processes a document using our new LlamaIndex extractor.
        This represents the data ingestion pipeline.
        
        Args:
            source_input: A file path or a file-like object from Streamlit.
            document_type: (No longer used) The schema is now fixed in the extractor.
        """
        # 1. Extract structured data
        print(f"Processing document: {getattr(source_input, 'name', source_input)}")
        
        try:
            # The extractor expects a list of sources
            extracted_results = await extract_documents([source_input])
            if not extracted_results:
                raise ValueError("Extraction returned no results.")
            
            # We only process one document at a time in this flow
            result = extracted_results[0]
            
            # Convert structured data to a single text block for chunking
            extracted_text = self._structured_data_to_text(result.get('data', {}))
            metadata = {
                "source": result.get('filename'),
                "document_type": "Income Statement",
                "structured_extraction": True,
            }

        except Exception as e:
            print(f"Error during document extraction: {e}")
            return {
                "processed_chunks": 0,
                "error": str(e),
                "extracted_text_length": 0
            }

        # 2. Chunk the extracted text
        print("Chunking text...")
        text_chunks = self.chunk_document(extracted_text)
        
        # Validate that we have chunks to process
        if not text_chunks:
            error_msg = "No text chunks were created. The document may be empty or extraction failed."
            print(f"ERROR: {error_msg}")
            return {
                "processed_chunks": 0,
                "error": error_msg,
                "extracted_text_length": len(extracted_text)
            }
        
        # Add metadata to chunks
        for chunk in text_chunks:
            chunk['metadata'] = {**metadata, **chunk.get('metadata', {})}
        
        # 3. Create embeddings and load into the vector store
        source_id = "text_input" if is_text_input else source_input
        return self.embed_and_load(text_chunks, source_id)
    
    def _structured_data_to_text(self, data: Dict[str, Any]) -> str:
        """Convert structured extraction data to text for chunking."""
        if not data:
            return ""
        text_parts = []
        for key, value in data.items():
            if value is not None:
                # Format field name nicely
                field_name = key.replace('_', ' ').title()
                text_parts.append(f"{field_name}: {value}")
        return "\n".join(text_parts)

    def _get_fallback_schema(self):
        """Returns a simple Pydantic schema for fallback text extraction."""
        from pydantic import BaseModel, Field
        
        class FallbackSchema(BaseModel):
            extracted_text: str = Field(
                description="The full text content of the document."
            )
        return FallbackSchema

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
        # Validate input
        if not chunks:
            error_msg = "No chunks provided for embedding"
            print(f"ERROR: {error_msg}")
            return {"processed_chunks": 0, "error": error_msg}
        
        # Create embeddings for each chunk
        print(f"Creating embeddings for {len(chunks)} chunks...")
        texts_to_embed = [chunk['text'] for chunk in chunks]
        
        if not texts_to_embed:
            error_msg = "No text found in chunks"
            print(f"ERROR: {error_msg}")
            return {"processed_chunks": 0, "error": error_msg}
        
        embeddings = create_embeddings(texts_to_embed, model=self.embedding_model)
        
        if not embeddings:
            error_msg = "Failed to create embeddings"
            print(f"ERROR: {error_msg}")
            return {"processed_chunks": 0, "error": error_msg}

        # Prepare vectors for Pinecone
        vectors_to_upsert = []
        doc_basename = os.path.basename(source_id)
        for i, chunk in enumerate(chunks):
            # Merge chunk metadata with base metadata
            chunk_metadata = chunk.get('metadata', {})
            metadata = {
                "text": chunk['text'],
                "filename": doc_basename,
                "chunk_number": i,
                **chunk_metadata  # Include any additional metadata from extraction
            }
            
            vectors_to_upsert.append({
                "id": f"doc_{doc_basename}_{i}",
                "values": embeddings[i],
                "metadata": metadata
            })

        # Final validation before upsert
        if not vectors_to_upsert:
            error_msg = "No vectors were prepared for upsert"
            print(f"ERROR: {error_msg}")
            return {"processed_chunks": 0, "error": error_msg}

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