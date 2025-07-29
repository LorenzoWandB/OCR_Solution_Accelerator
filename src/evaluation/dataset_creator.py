import weave
import json
import os
from typing import List, Dict, Any
import random
from src.rag.chunker import chunk_text_with_overlap
from src.rag.embed import create_embeddings
from src.rag.vectore_store import PineconeVectorStore

def generate_synthetic_financial_documents() -> List[Dict[str, Any]]:
    """Generate synthetic financial documents with realistic data."""
    print("Generating synthetic financial dataset...")
    
    # Template financial documents
    synthetic_documents = [
        {
            'id': 'income_stmt_1',
            'text': """
            ANNUAL INCOME STATEMENT
            Company: TechCorp Inc.
            Year Ended December 31, 2023
            
            Revenue: $15,750,000
            Cost of Goods Sold: $8,250,000
            Gross Profit: $7,500,000
            
            Operating Expenses:
            - Sales and Marketing: $2,100,000
            - Research and Development: $1,800,000
            - General and Administrative: $950,000
            Total Operating Expenses: $4,850,000
            
            Operating Income: $2,650,000
            Interest Expense: $125,000
            Income Before Taxes: $2,525,000
            Income Tax Expense: $631,250
            Net Income: $1,893,750
            
            Earnings Per Share: $3.15
            """,
            'metadata': {'type': 'income_statement', 'year': 2023}
        },
        {
            'id': 'balance_sheet_1', 
            'text': """
            BALANCE SHEET
            Company: TechCorp Inc.
            As of December 31, 2023
            
            ASSETS
            Current Assets:
            - Cash and Cash Equivalents: $2,850,000
            - Accounts Receivable: $3,200,000
            - Inventory: $1,450,000
            - Prepaid Expenses: $285,000
            Total Current Assets: $7,785,000
            
            Property, Plant & Equipment: $8,920,000
            Intangible Assets: $2,340,000
            Total Assets: $19,045,000
            
            LIABILITIES & EQUITY
            Current Liabilities:
            - Accounts Payable: $1,680,000
            - Accrued Expenses: $745,000
            - Short-term Debt: $500,000
            Total Current Liabilities: $2,925,000
            
            Long-term Debt: $4,200,000
            Total Liabilities: $7,125,000
            
            Stockholders' Equity: $11,920,000
            Total Liabilities & Equity: $19,045,000
            """,
            'metadata': {'type': 'balance_sheet', 'year': 2023}
        },
        {
            'id': 'cash_flow_1',
            'text': """
            CASH FLOW STATEMENT
            Company: TechCorp Inc.
            Year Ended December 31, 2023
            
            OPERATING ACTIVITIES
            Net Income: $1,893,750
            Depreciation: $890,000
            Changes in Working Capital: ($235,000)
            Cash Flow from Operations: $2,548,750
            
            INVESTING ACTIVITIES
            Capital Expenditures: ($1,250,000)
            Equipment Purchases: ($340,000)
            Cash Flow from Investing: ($1,590,000)
            
            FINANCING ACTIVITIES
            Dividend Payments: ($420,000)
            Stock Repurchases: ($180,000)
            Cash Flow from Financing: ($600,000)
            
            Net Change in Cash: $358,750
            Beginning Cash Balance: $2,491,250
            Ending Cash Balance: $2,850,000
            """,
            'metadata': {'type': 'cash_flow', 'year': 2023}
        }
    ]
    
    # Generate additional synthetic documents with varying numbers
    for i in range(4):  # Create 3 more companies
        for year in [2021, 2022, 2023]:
            company_num = i
            revenue = random.randint(10000000, 50000000)
            gross_margin = random.uniform(0.35, 0.65)
            assets = random.randint(15000000, 80000000)
            cash_flow = random.randint(1000000, 8000000)
            
            synthetic_documents.append({
                'id': f'synthetic_{company_num}_{year}',
                'text': f"""
                FINANCIAL HIGHLIGHTS
                Company: InnovateCorp {company_num}
                Year: {year}
                
                Key Metrics:
                - Revenue: ${revenue:,}
                - Gross Margin: {gross_margin:.1%}
                - Total Assets: ${assets:,}
                - Operating Cash Flow: ${cash_flow:,}
                - Market Cap: ${assets * 1.5:,.0f}
                
                Business Overview:
                Leading technology company focused on innovative solutions.
                Strong financial performance with consistent growth trajectory.
                """,
                'metadata': {'type': 'financial_highlights', 'year': year, 'company': f'InnovateCorp_{company_num}'}
            })
    
    return synthetic_documents

def create_qa_pairs_from_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create question-answer pairs from financial documents."""
    qa_pairs = []
    qa_id = 0
    
    # Standard financial questions to ask for each document
    financial_questions = [
        "What is the revenue?",
        "What is the net income?", 
        "What is the gross profit?",
        "What are the total operating expenses?",
        "What is the earnings per share?",
        "What are the total assets?",
        "What is the cash balance?",
        "What is the total debt?",
        "What is the total equity?",
        "What is the accounts receivable?",
        "What is the cash flow from operations?",
        "What is the ending cash balance?",
        "What is the net change in cash?",
        "How much was spent on equipment?",
        "What were the dividend payments?",
        "What is the gross margin?",
        "What are the total assets?",
        "What is the cash flow from operations?"
    ]
    
    for doc in documents:
        # Create multiple QA pairs per document
        questions_for_doc = random.sample(financial_questions, min(5, len(financial_questions)))
        
        for question in questions_for_doc:
            qa_pairs.append({
                'id': f"{doc['id']}_qa_{qa_id}",
                'query': question,
                'document_id': doc['id'],
                'relevant_context': [doc['text']],  # The document text is the relevant context
                'metadata': {
                    'source_document': doc['id'],
                    'question_type': 'financial_metric',
                    **doc['metadata']
                }
            })
            qa_id += 1
    
    return qa_pairs

@weave.op()
def create_synthetic_evaluation_dataset(index_name: str, namespace: str = "eval_dataset") -> Dict[str, Any]:
    """
    Create a complete synthetic evaluation dataset and store it in both Weave and Pinecone.
    
    Args:
        index_name: Pinecone index name
        namespace: Pinecone namespace for evaluation data
        
    Returns:
        Dataset creation summary
    """
    # Initialize vector store
    vector_store = PineconeVectorStore(index_name=index_name, namespace=namespace)
    vector_store.initialize_index(dimension=1536)
    
    # Step 1: Generate synthetic documents
    documents = generate_synthetic_financial_documents()
    
    # Step 2: Create QA pairs
    qa_pairs = create_qa_pairs_from_documents(documents)
    
    # Step 3: Chunk documents for vector storage
    print(f"Chunking {len(documents)} documents...")
    chunked_documents = []
    for doc in documents:
        chunks = chunk_text_with_overlap(doc['text'], chunk_size=500, overlap=100)
        for i, chunk_dict in enumerate(chunks):
            chunked_documents.append({
                'id': f"{doc['id']}_chunk_{i}",
                'text': chunk_dict['text'],  # Extract text from the chunk dictionary
                'original_doc_id': doc['id'],
                'chunk_index': i,
                'metadata': doc['metadata']
            })
    
    # Step 4: Create embeddings and store in Pinecone
    print(f"Creating embeddings for {len(chunked_documents)} chunks...")
    chunk_texts = [chunk['text'] for chunk in chunked_documents]
    embeddings = create_embeddings(chunk_texts)
    
    # Prepare vectors for Pinecone
    vectors_to_upsert = []
    for i, chunk in enumerate(chunked_documents):
        vectors_to_upsert.append({
            "id": chunk['id'],
            "values": embeddings[i],
            "metadata": {
                "text": chunk['text'],
                "original_doc_id": chunk['original_doc_id'],
                "chunk_index": chunk['chunk_index'],
                **chunk['metadata']
            }
        })
    
    # Batch upsert to Pinecone
    print(f"Storing {len(vectors_to_upsert)} vectors in Pinecone...")
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        vector_store.upsert(batch)
    
    print(f"Upserted {len(vectors_to_upsert)} vectors to namespace '{namespace}'.")
    
    # Step 5: Create Weave dataset
    print("Creating Weave dataset...")
    dataset_rows = []
    for qa in qa_pairs:
        dataset_rows.append({
            'id': qa['id'],
            'query': qa['query'],
            'document_id': qa['document_id'],
            'relevant_context': qa['relevant_context'],
            'metadata': qa['metadata']
        })
    
    # Create and publish Weave dataset
    dataset = weave.Dataset(
        name="financial_rag_eval_synthetic",
        rows=dataset_rows
    )
    dataset_ref = weave.publish(dataset)
    
    print(f"📦 Published to {dataset_ref}")
    
    return {
        'dataset_ref': dataset_ref,
        'dataset_ref_str': str(dataset_ref),
        'total_documents': len(documents),
        'total_chunks': len(chunked_documents),
        'total_qa_pairs': len(qa_pairs),
        'pinecone_namespace': namespace,
        'dataset_source': 'synthetic'
    } 