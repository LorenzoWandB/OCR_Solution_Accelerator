from src.rag.vectore_store import PineconeVectorStore
from src.rag.embed import create_embeddings

class Retriever:
    def __init__(self, index_name: str):
        self.vector_store = PineconeVectorStore(index_name=index_name)
        self.vector_store.initialize_index(dimension=1536)

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

        return results.get('matches', [])

    def retrieve_and_rerank(self, query: str, top_k: int = 10, rerank_top_n: int = 3):
        """
        Retrieves documents and then reranks them to improve relevance.
        """
        print(f"Retrieving top {top_k} documents for reranking...")
        initial_results = self.retrieve(query, top_k=top_k)

        if not initial_results:
            return []

        # Extract the text from the metadata to prepare for reranking
        docs_to_rerank = [match['metadata']['text'] for match in initial_results]

        # Use Pinecone's rerank API
        print("Reranking results...")
        reranked_results = self.vector_store.pc.inference.rerank(
            model="pinecone-rerank-v0",
            query=query,
            documents=docs_to_rerank,
            top_n=rerank_top_n
        )

        return reranked_results.get('results', [])
