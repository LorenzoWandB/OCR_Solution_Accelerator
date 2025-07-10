import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import weave

load_dotenv()

@weave.op()
class PineconeVectorStore:
    def __init__(self, index_name: str, api_key: str = None, namespace: str = None):
        if api_key is None:
            api_key=os.environ.get("PINECONE_API_KEY")
        
        if not api_key or not index_name:
            raise ValueError("Pinecone API key and index name are required.")

        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.namespace = namespace if namespace else None
        self.index = None
    # Create a serverless index if it doesn't exist and connect to it
    @weave.op()
    def initialize_index(self, dimension: int, metric: str = 'cosine', cloud: str = 'aws', region:str = 'us-east-1'):
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )
            print(f"Index '{self.index_name}' created successfully.")
        
        self.index = self.pc.Index(self.index_name)
        print(f"Connected to index '{self.index_name}'.")

    # Upsert vectors into the Pinecone index
    @weave.op()
    def upsert(self, vectors, namespace: str = None):
        if not self.index:
            raise ConnectionError("Index is not initialized. Call initialize_index() first.")
        
        namespace_to_use = namespace if namespace is not None else self.namespace
        self.index.upsert(vectors=vectors, namespace=namespace_to_use)
        print(f"Upserted {len(vectors)} vectors to namespace '{namespace_to_use or 'default'}'.")

    # Query the Pinecone index for similar vectors
    @weave.op()
    def query(self, vector, top_k: int = 5, namespace: str = None, include_metadata: bool = True):
        if not self.index:
            raise ConnectionError("Index is not initialized. Call initialize_index() first.")
            
        namespace_to_use = namespace if namespace is not None else self.namespace
        return self.index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace_to_use,
            include_metadata=include_metadata
        )
    
    # Get statistics about the index
    @weave.op()
    def describe_index_stats(self):
        if not self.index:
            raise ConnectionError("Index is not initialized. Call initialize_index() first.")
        
        return self.index.describe_index_stats()
