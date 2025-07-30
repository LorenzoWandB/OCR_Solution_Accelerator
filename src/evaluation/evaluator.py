import weave
from src.evaluation.dataset_creator import create_synthetic_evaluation_dataset

def create_evaluation_dataset(index_name: str, 
                            namespace: str = "eval_dataset"):
    """
    Create and return a synthetic Weave dataset for evaluation.
    
    Args:
        index_name: Pinecone index name
        namespace: Pinecone namespace for evaluation data
        
    Returns:
        Weave dataset object
    """
    # Initialize Weave if not already initialized
    try:
        import weave
        from weave.trace.context import weave_client_context
        weave_client_context.require_weave_client()
    except:
        weave.init("Solution-Accelerator-MRM-Eval")
    
    result = create_synthetic_evaluation_dataset(index_name, namespace)
    # Return the dataset object directly for simpler handling
    dataset_object = result['dataset_object']
    return dataset_object 