import weave
import json
import logging
from src.rag.vectore_store import PineconeVectorStore
from src.rag.embed import create_embeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, index_name: str, namespace: str = None):
        self.vector_store = PineconeVectorStore(index_name=index_name, namespace=namespace)
        self.vector_store.initialize_index(dimension=1536)
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess the query for better matching.
        """
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Convert to lowercase for consistency
        query = query.lower()
        
        # Remove common stop words that might not add value
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had'}
        words = query.split()
        
        # Keep stop words if query is short
        if len(words) > 5:
            words = [w for w in words if w not in stop_words or len(w) > 3]
            query = ' '.join(words)
        
        return query.strip()

    @weave.op()
    def retrieve(self, query: str, top_k: int = 5, namespace: str = None, preprocess: bool = True):
        logger.info(f"Retrieving documents for query: '{query}' with top_k={top_k}")
        
        try:
            # Optionally preprocess the query
            processed_query = self.preprocess_query(query) if preprocess else query
            if processed_query != query:
                logger.debug(f"Query preprocessed: '{query}' -> '{processed_query}'")
            
            query_embedding = create_embeddings(processed_query)
            if query_embedding is None:
                logger.error("Could not create embedding for the query.")
                return []

            logger.debug(f"Query embedding created successfully")
            
            results = self.vector_store.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace or self.vector_store.namespace
            )

            if not results or not results.get('matches'):
                logger.warning(f"No results found for query: '{query}'")
                return []

            matches = results.get('matches', [])
            logger.info(f"Found {len(matches)} matches for query: '{query}'")
            
            # Log sample of results for debugging
            if matches:
                logger.debug(f"Top match score: {matches[0].get('score', 0):.4f}")
            
            return matches
            
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            return []

    def retrieve_and_rerank(self, query: str, top_k: int = 10, rerank_top_n: int = 3, namespace: str = None):
        """
        Retrieves documents and then reranks them to improve relevance.
        """
        logger.info(f"Retrieving top {top_k} documents for reranking...")
        
        try:
            initial_results = self.retrieve(query, top_k=top_k, namespace=namespace)

            if not initial_results:
                return []

            # Prepare documents for reranking, keeping track of original metadata
            docs_for_reranking = []
            original_docs_map = {}
            
            for i, match in enumerate(initial_results):
                metadata = match.get('metadata', {})
                text_content = metadata.get('text', '')
                
                if not text_content:
                    logger.warning(f"Empty text content in match {i}")
                    continue
                    
                docs_for_reranking.append(text_content)
                original_docs_map[i] = match

            if not docs_for_reranking:
                logger.warning("No valid documents for reranking")
                return initial_results[:rerank_top_n]

            # Use Pinecone's rerank API
            logger.info(f"Reranking {len(docs_for_reranking)} results...")
            reranked_results = self.vector_store.pc.inference.rerank(
                model="pinecone-rerank-v0",
                query=query,
                documents=docs_for_reranking,
                top_n=min(rerank_top_n, len(docs_for_reranking))
            )

            # Handle empty rerank results
            if not reranked_results or not hasattr(reranked_results, 'results'):
                logger.warning("Reranking failed, returning initial results")
                return initial_results[:rerank_top_n]

            # Re-associate reranked results with original metadata
            formatted_results = []
            for result in reranked_results.results:
                original_index = result.index
                original_match = original_docs_map.get(original_index)
                
                if original_match:
                    # Create a copy to avoid modifying the original
                    enhanced_match = original_match.copy()
                    enhanced_match['score'] = result.relevance_score
                    formatted_results.append(enhanced_match)
                    
            logger.info(f"Reranking complete. Returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            logger.info("Falling back to initial results")
            return initial_results[:rerank_top_n] if initial_results else []
