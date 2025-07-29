import weave
from dotenv import load_dotenv
import os
import argparse
from src.weave.model import RagModel
from src.evaluation.evaluator import create_evaluation_dataset
from src.evaluation.weave_native_eval import run_weave_native_evaluation_sync

load_dotenv()

# --- Configuration ---
PROJECT = "Solution-Accelerator-MRM"
PINECONE_INDEX_NAME = "mrm-rag-weave" # Using a new index for clarity
PINECONE_NAMESPACE = "financial_statements"
IMAGE_PATH = "data/images/simple_statement.jpg"

def create_dataset(args):
    """Create synthetic evaluation dataset and store in Weave and Pinecone."""
    print("Creating synthetic evaluation dataset...")
    
    # Use different index/namespace for evaluation
    eval_index_name = f"{PINECONE_INDEX_NAME}-eval"
    eval_namespace = "eval_dataset"
    
    dataset_ref = create_evaluation_dataset(
        index_name=eval_index_name,
        namespace=eval_namespace
    )
    
    print(f"✓ Synthetic dataset created successfully!")
    print(f"  Dataset reference: {dataset_ref}")
    print(f"  Pinecone index: {eval_index_name}")
    print(f"  Namespace: {eval_namespace}")

def run_evaluation(args):
    """Run evaluation using Weave's native evaluation system."""
    print("Setting up Weave native evaluation...")
    
    # Initialize Weave for evaluation
    weave.init("Solution-Accelerator-MRM-Eval")
    
    # Create dataset if requested
    dataset_ref = args.dataset_ref
    if args.create_dataset or not dataset_ref:
        print("Creating synthetic evaluation dataset...")
        dataset_ref = create_evaluation_dataset(
            index_name=f"{PINECONE_INDEX_NAME}-eval",
            namespace="eval_dataset"
        )
        print(f"✓ Synthetic dataset created: {dataset_ref}")
    
    # Run evaluation using Weave's native system
    # Use the same index and namespace where evaluation data is stored
    eval_index_name = f"{PINECONE_INDEX_NAME}-eval"
    eval_namespace = "eval_dataset"
    
    results = run_weave_native_evaluation_sync(
        index_name=eval_index_name,
        namespace=eval_namespace,
        dataset_ref=dataset_ref,
        k=5
    )
    
    # Print summary
    evaluation_results = results.get("evaluation_results", {})
    print("\n" + "="*50)
    print("WEAVE NATIVE EVALUATION SUMMARY")
    print("="*50)
    print(f"Model: {results['model_config']}")
    print(f"Dataset: {str(dataset_ref)}")
    print(f"Evaluation Results: {evaluation_results}")
    
    # Save results if requested
    if args.save_results:
        results_file = "weave_evaluation_results_synthetic.json"
        import json
        with open(results_file, 'w') as f:
            # Convert to JSON-serializable format
            serializable_results = {
                "model_config": results["model_config"],
                "dataset_ref": str(dataset_ref),
                "evaluation_summary": str(evaluation_results)
            }
            json.dump(serializable_results, f, indent=2)
        print(f"\n✓ Results saved to: {results_file}")
    
    return results

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
        ingestion_result = rag_model.load_and_process_document(IMAGE_PATH, is_text_input=False)
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
    parser = argparse.ArgumentParser(description="Financial RAG Model with Evaluation")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Default command (original main functionality)
    parser_main = subparsers.add_parser('run', help='Run the main RAG pipeline (default)')
    
    # Dataset creation command
    parser_dataset = subparsers.add_parser('create-dataset', help='Create synthetic evaluation dataset')
    
    # Evaluation command
    parser_eval = subparsers.add_parser('evaluate', help='Run comprehensive evaluation')
    parser_eval.add_argument(
        '--dataset-ref',
        help='Weave dataset reference for evaluation'
    )
    parser_eval.add_argument(
        '--create-dataset',
        action='store_true',
        help='Create new dataset before evaluation'
    )

    parser_eval.add_argument(
        '--save-results',
        action='store_true',
        help='Save evaluation results to JSON file'
    )
    
    args = parser.parse_args()
    
    # If no command specified, run main
    if args.command is None or args.command == 'run':
        main()
    elif args.command == 'create-dataset':
        create_dataset(args)
    elif args.command == 'evaluate':
        run_evaluation(args)
    else:
        parser.print_help()



