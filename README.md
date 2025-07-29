# RAG System for Financial Document Processing

A Retrieval-Augmented Generation (RAG) system that processes image-based financial documents using OCR and semantic search.

## Architecture Overview

**Document Processing Pipeline:**
1. **OCR Extraction**: OpenAI GPT-4.1 vision model converts images to text
2. **Text Chunking**: Adaptive chunking (line-by-line, overlapping, or paragraph-based)
3. **Embedding Generation**: OpenAI `text-embedding-3-small` (1536 dimensions)
4. **Vector Storage**: Pinecone vector database with cosine similarity

**RAG Retrieval Flow:**
1. Query embedding generation
2. Vector similarity search in Pinecone
3. Optional reranking using Pinecone rerank API
4. Return ranked results for generation

## Key Components

- **OCR Extractor** (`src/ocr/extractor.py`): Image-to-text conversion
- **Chunker** (`src/rag/chunker.py`): Intelligent text segmentation
- **Embedder** (`src/rag/embed.py`): OpenAI embedding integration
- **Vector Store** (`src/rag/vectore_store.py`): Pinecone database management
- **Retriever** (`src/rag/retriever.py`): Semantic search with reranking

## User Interfaces

- **CLI** (`main.py`): Complete pipeline demonstration
- **Streamlit App** (`streamlit_app.py`): Web interface for upload, processing, and search

## Technology Stack

- **Vector Database**: Pinecone (serverless, AWS us-east-1)
- **Embeddings**: OpenAI text-embedding-3-small
- **OCR**: OpenAI GPT-4.1 vision
- **Reranking**: Pinecone rerank-v0
- **Monitoring**: Weights & Biases Weave
- **UI**: Streamlit

## Setup & Installation

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file with:
```bash
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_environment
WANDB_API_KEY=your_wandb_key
```

## Evaluation System

The system includes a comprehensive evaluation framework for financial RAG applications using **W&B Weave**.

### 📊 Financial-Specific Metrics

**Retrieval Metrics:**
- **Recall@k**: Fraction of relevant documents retrieved in top-k results
- **MRR@k**: Mean Reciprocal Rank - how high the first relevant document appears

**Generation Metrics:**
- **Faithfulness**: Ensures answers are grounded in retrieved context (using Weave's evaluation system)
- **Numeric Consistency**: Validates all numbers in answers appear in source context (critical for financial accuracy)

### 🗃️ Evaluation Dataset

The system uses **synthetic financial documents** including:
- **Income Statements** (revenue, expenses, net income)
- **Balance Sheets** (assets, liabilities, equity)  
- **Cash Flow Statements** (operating, investing, financing activities)
- **Financial Highlights** (key metrics across multiple companies/years)

**Dataset Features:**
- 63 Q&A pairs covering typical financial queries
- Realistic financial formatting ($, commas, percentages)
- Multiple document types and companies
- Stored as `weave.Dataset` for reproducibility

### 🚀 Quick Start - Evaluation

#### First-Time Setup (Generate Dataset)
```bash
# Generate synthetic financial dataset and run evaluation
python main.py evaluate --create-dataset --save-results
```

#### Subsequent Evaluations
```bash
# Run evaluation on existing dataset
python main.py evaluate --dataset-ref "your-dataset-reference" --save-results
```

#### Dataset-Only Creation
```bash
# Create synthetic dataset without running evaluation
python main.py create-dataset
```

### 📈 Understanding Results

**Example Output:**
```
WEAVE NATIVE EVALUATION SUMMARY
==================================================
✅ Evaluated: 63/63 samples
⏱️ Avg Latency: 2.14s per query

Metrics:
- Recall@k: 0.85 (85% relevant docs retrieved)
- MRR@k: 0.67 (first relevant doc avg rank: 1.5)
- Numeric Consistency: 79.4% (excellent for finance!)  
- Faithfulness: 42.7% (room for improvement)
```

**Key Thresholds:**
- **Numeric Consistency >90%**: Essential for financial applications
- **Recall@k >70%**: Good retrieval performance
- **Faithfulness >80%**: Low hallucination risk

### 📊 Monitoring & Analysis

- **Live Dashboard**: https://wandb.ai/your-project/weave
- **Results Files**: Saved as JSON for further analysis
- **Weave Traces**: Full traceability of each evaluation step

## Basic Usage (RAG Pipeline)

### CLI Interface
```bash
# Run basic RAG pipeline on financial document
python main.py run
```

### Web Interface
```bash
# Launch Streamlit app
streamlit run streamlit_app.py
```

Focus on financial document processing with example queries like "What is the Net Profit?" for income statements and financial reports.