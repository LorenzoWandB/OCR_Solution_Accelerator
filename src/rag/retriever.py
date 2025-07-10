import weave
import json
from src.rag.vectore_store import PineconeVectorStore
from src.rag.embed import create_embeddings

class Retriever:
    def __init__(self, index_name: str, namespace: str = None):
        self.vector_store = PineconeVectorStore(index_name=index_name, namespace=namespace)
        self.vector_store.initialize_index(dimension=1536)

    @weave.op()
    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = create_embeddings(query)
        if query_embedding is None:
            print("Could not create embedding for the query.")
            return []

        results = self.vector_store.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        if not results or not results['matches']:
            print("No results found.")
            return []

        return results.get('matches', [])

    def retrieve_and_rerank(self, query: str, top_k: int = 10, rerank_top_n: int = 3):
        """
        Retrieves documents and then reranks them to improve relevance.
        """
        print(f"Retrieving top {top_k} documents for reranking...")
        initial_results = self.retrieve(query, top_k=top_k)

        if not initial_results:
            return []

        # Prepare documents for reranking. We pass the metadata from the initial results.
        # Also, we need to extract the text from the JSON string in the 'text' field.
        docs_for_reranking = []
        for match in initial_results:
            metadata = match['metadata']
            try:
                # The 'text' field contains a JSON string, so we parse it.
                doc_content = json.loads(metadata['text'])
                # We create a single string from the dictionary values for the reranker.
                clean_text = " ".join(str(v) for v in doc_content.values())
                docs_for_reranking.append(clean_text)
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                print(f"Skipping document due to error processing text field: {e}")
                # If parsing fails, we can fall back to using the raw text or skip.
                # Here we'll just use the raw text if it's a string.
                if isinstance(metadata.get('text'), str):
                    docs_for_reranking.append(metadata['text'])


        # Use Pinecone's rerank API
        print("Reranking results...")
        reranked_results = self.vector_store.pc.inference.rerank(
            model="pinecone-rerank-v0",
            query=query,
            documents=docs_for_reranking,
            top_n=rerank_top_n,
            return_documents=True
        )

        # Format results to match expected structure in main.py
        formatted_results = []
        for result in reranked_results.data:
            formatted_results.append({
                'score': result['score'],
                'document': result['document']
            })

        return formatted_results
