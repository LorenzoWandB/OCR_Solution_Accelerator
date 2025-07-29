import weave
from src.evaluation.dataset_creator import create_synthetic_evaluation_dataset

def create_evaluation_dataset(index_name: str, 
                            namespace: str = "eval_dataset") -> str:
    """
    Create and return a synthetic Weave dataset for evaluation.
    
    Args:
        index_name: Pinecone index name
        namespace: Pinecone namespace for evaluation data
        
    Returns:
        Weave dataset reference
    """
    # Initialize Weave if not already initialized
    try:
        import weave
        from weave.trace.context import weave_client_context
        weave_client_context.require_weave_client()
    except:
        weave.init("Solution-Accelerator-MRM-Eval")
    
    result = create_synthetic_evaluation_dataset(index_name, namespace)
    # Return the ObjectRef directly for Weave evaluation
    dataset_ref = result['dataset_ref']
    return dataset_ref 