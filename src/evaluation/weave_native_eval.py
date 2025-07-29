import weave
from weave import Evaluation, Model
import asyncio
import re
import numpy as np
from typing import Dict, Any, List, Optional
from src.weave.model import RagModel

# === CORE METRIC FUNCTIONS ===

@weave.op()
def recall_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: Optional[int] = None) -> float:
    """Calculate Recall@k - fraction of relevant documents retrieved in top-k results."""
    if not relevant_docs:
        return 1.0  # No relevant docs to retrieve
    
    if not retrieved_docs:
        return 0.0  # No docs retrieved
    
    # Consider only top-k retrieved documents
    top_k_retrieved = retrieved_docs[:k] if k is not None else retrieved_docs
    
    # Count how many relevant docs are in the top-k retrieved
    retrieved_relevant = 0
    for doc in top_k_retrieved:
        if any(relevant_doc in doc or doc in relevant_doc for relevant_doc in relevant_docs):
            retrieved_relevant += 1
    
    return retrieved_relevant / len(relevant_docs)

@weave.op()
def mrr_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: Optional[int] = None) -> float:
    """Calculate Mean Reciprocal Rank@k - reciprocal of rank of first relevant document."""
    if not relevant_docs or not retrieved_docs:
        return 0.0
    
    # Consider only top-k retrieved documents
    top_k_retrieved = retrieved_docs[:k] if k is not None else retrieved_docs
    
    # Find the rank of the first relevant document
    for rank, doc in enumerate(top_k_retrieved, 1):
        if any(relevant_doc in doc or doc in relevant_doc for relevant_doc in relevant_docs):
            return 1.0 / rank
    
    return 0.0  # No relevant documents found

@weave.op()
def numeric_consistency(generated_answer: str, context: List[str]) -> Dict[str, Any]:
    """Check if all numbers in generated answer appear in context - crucial for financial accuracy."""
    # Extract numbers using regex (integers, decimals, percentages, currency)
    number_pattern = r'(?:[$£€¥]?\s*)?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:\s*[%$£€¥])?'
    answer_numbers = re.findall(number_pattern, generated_answer)
    answer_numbers = [num.strip() for num in answer_numbers if num.strip()]
    
    if not answer_numbers:
        return {
            "consistency_score": 1.0,  # No numbers to check
            "total_numbers": 0,
            "consistent_numbers": 0,
            "inconsistent_numbers": [],
            "details": "No numbers found in generated answer"
        }
    
    def normalize_number(num_str: str) -> str:
        """Remove currency symbols and spaces for comparison."""
        return re.sub(r'[$£€¥%\s]', '', num_str).lower()
    
    # Extract numbers from context
    context_text = " ".join(context)
    context_numbers = re.findall(number_pattern, context_text)
    context_numbers_normalized = [normalize_number(num) for num in context_numbers]
    
    # Check consistency
    consistent_numbers = 0
    inconsistent_numbers = []
    
    for answer_num in answer_numbers:
        answer_num_normalized = normalize_number(answer_num)
        if answer_num_normalized in context_numbers_normalized:
            consistent_numbers += 1
        else:
            inconsistent_numbers.append(answer_num)
    
    consistency_score = consistent_numbers / len(answer_numbers)
    
    return {
        "consistency_score": consistency_score,
        "total_numbers": len(answer_numbers),
        "consistent_numbers": consistent_numbers,
        "inconsistent_numbers": inconsistent_numbers,
        "details": f"Found {len(answer_numbers)} numbers, {consistent_numbers} consistent"
    }

@weave.op()
def simple_faithfulness_scorer(output: Dict[str, Any]) -> Dict[str, float]:
    """Enhanced faithfulness check - ensures answer is grounded in retrieved context."""
    answer = output.get('generated_answer', '').lower()
    context = output.get('retrieved_context', [])
    
    if not answer or not context:
        return {"faithfulness_score": 0.0}
    
    context_text = " ".join(context).lower()
    
    # Remove common stop words and punctuation for better matching
    import re
    def clean_text(text):
        # Remove punctuation and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out very short words and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had'}
        return set(word for word in words if len(word) > 2 and word not in stop_words)
    
    answer_words = clean_text(answer)
    context_words = clean_text(context_text)
    
    if not answer_words:
        return {"faithfulness_score": 0.0}
    
    # Calculate overlap ratio - focus on meaningful words
    overlap = len(answer_words.intersection(context_words))
    faithfulness_score = overlap / len(answer_words)
    
    # Give bonus for numerical overlaps (important for financial data)
    numbers_in_answer = set(re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', answer))
    numbers_in_context = set(re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', context_text))
    
    if numbers_in_answer:
        number_overlap = len(numbers_in_answer.intersection(numbers_in_context))
        number_bonus = (number_overlap / len(numbers_in_answer)) * 0.3  # 30% bonus weight
        faithfulness_score = min(faithfulness_score + number_bonus, 1.0)
    
    return {"faithfulness_score": faithfulness_score}

# === WEAVE SCORERS ===

@weave.op()
def recall_at_k_scorer(relevant_context: List[str], output: Dict[str, Any]) -> Dict[str, float]:
    """Weave scorer for Recall@k metric."""
    retrieved_docs = output.get('retrieved_context', [])
    relevant_docs = relevant_context
    
    k = len(retrieved_docs)  # Use actual number of retrieved docs
    score = recall_at_k(retrieved_docs, relevant_docs, k)
    return {f'recall_at_{k}': score}

@weave.op()
def mrr_at_k_scorer(relevant_context: List[str], output: Dict[str, Any]) -> Dict[str, float]:
    """Weave scorer for MRR@k metric."""
    retrieved_docs = output.get('retrieved_context', [])
    relevant_docs = relevant_context
    
    k = len(retrieved_docs)  # Use actual number of retrieved docs
    score = mrr_at_k(retrieved_docs, relevant_docs, k)
    return {f'mrr_at_{k}': score}

@weave.op()
def numeric_consistency_scorer(output: Dict[str, Any]) -> Dict[str, Any]:
    """Weave scorer for numeric consistency."""
    answer = output.get('generated_answer', '')
    context = output.get('retrieved_context', [])
    
    metrics = numeric_consistency(answer, context)
    return {
        'numeric_consistency_score': metrics['consistency_score'],
        'numeric_details': {
            'consistency_score': metrics['consistency_score'],
            'total_numbers': metrics['total_numbers'],
            'consistent_numbers': metrics['consistent_numbers']
        }
    }

# === WEAVE EVALUATION SETUP ===

def create_weave_evaluation(dataset_ref, k: int = 5) -> Evaluation:
    """Create a Weave Evaluation object using the official API."""
    # Load the dataset - handle both ObjectRef and string formats
    try:
        if hasattr(dataset_ref, 'get'):
            # It's an ObjectRef, use it directly
            dataset = dataset_ref.get()
            print(f"✓ Loaded dataset from ObjectRef: {dataset_ref}")
        else:
            # It's a string, convert to ref
            dataset = weave.ref(dataset_ref).get()
            print(f"✓ Loaded dataset from string ref: {dataset_ref}")
    except Exception as e:
        print(f"Error loading dataset {dataset_ref}: {e}")
        raise e
    
    print(f"Running evaluation on dataset: {dataset_ref}")
    
    # Create evaluation with all scorers
    evaluation = Evaluation(
        dataset=dataset,
        scorers=[
            recall_at_k_scorer,
            mrr_at_k_scorer, 
            numeric_consistency_scorer,
            simple_faithfulness_scorer
        ],
        preprocess_model_input=lambda x: {"query": x["query"]}  # Return dict with query
    )
    
    return evaluation

async def run_weave_native_evaluation(
    index_name: str = "mrm-rag-weave",
    namespace: str = "financial_statements", 
    dataset_ref = None,
    k: int = 5
) -> Dict[str, Any]:
    """Run evaluation using Weave's native evaluation system."""
    print("Setting up Weave native evaluation...")
    
    # Create the model for evaluation
    model = RagModel(
        index_name=index_name,
        namespace=namespace,
        embedding_model="text-embedding-3-small",
        chunk_size=500,
        chunk_overlap=100,
        retriever_top_k=k
    )
    
    print(f"Model configuration: index={index_name}, namespace={namespace}, k={k}")
    
    # Create evaluation
    evaluation = create_weave_evaluation(dataset_ref, k)
    
    # Run evaluation
    evaluation_results = await evaluation.evaluate(model)
    
    print("✅ Weave evaluation completed!")
    
    return {
        "evaluation_results": evaluation_results,
        "model_config": {
            "index_name": index_name,
            "namespace": namespace,
            "k": k
        },
        "dataset_ref": dataset_ref
    }

def run_weave_native_evaluation_sync(
    index_name: str = "mrm-rag-weave",
    namespace: str = "financial_statements", 
    dataset_ref = None,
    k: int = 5
) -> Dict[str, Any]:
    """Synchronous wrapper for the async evaluation function."""
    return asyncio.run(run_weave_native_evaluation(
        index_name=index_name,
        namespace=namespace,
        dataset_ref=dataset_ref,
        k=k
    )) 