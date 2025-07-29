# Financial RAG System with Evaluation

A Retrieval-Augmented Generation (RAG) system for financial document processing with comprehensive evaluation using **W&B Weave**.

## 🏗️ Architecture

**Pipeline:**
1. **OCR**: OpenAI GPT-4 vision → text extraction
2. **Chunking**: Intelligent text segmentation  
3. **Embedding**: OpenAI `text-embedding-3-small` (1536D)
4. **Storage**: Pinecone vector database
5. **Retrieval**: Semantic search + GPT-4o generation

## 🚀 Quick Start

### Setup
```bash
# Virtual environment (required)
python -m venv venv
source venv/bin/activate

# Install dependencies  
pip install -r requirements.txt
```

### Environment Variables
Create `.env` file:
```bash
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_environment
WANDB_API_KEY=your_wandb_key
```

## 📊 Evaluation System

### Financial-Specific Metrics
- **Recall@5**: Fraction of relevant documents retrieved (26.7%)
- **MRR@5**: Mean Reciprocal Rank of first relevant doc (10.7%)  
- **Numeric Consistency**: Financial numbers grounded in source (97.5%)
- **Faithfulness**: Answer grounded in context (85.2%)

### Dataset
- **75 synthetic Q&A pairs** covering financial statements
- Income statements, balance sheets, cash flow statements
- Realistic financial formatting and multi-company data
- Stored as `weave.Dataset` for reproducibility

### Run Evaluation

**First time (creates dataset):**
```bash
python main.py evaluate --create-dataset --save-results
```

**Subsequent runs:**
```bash
python main.py evaluate --save-results
```

**Dataset creation only:**
```bash
python main.py create-dataset
```

### Results Dashboard
- **Live tracking**: https://wandb.ai/your-project/weave  
- **JSON results**: `weave_evaluation_results_synthetic.json`
- **Full traceability** via Weave traces

## 💻 Basic Usage

**CLI Pipeline:**
```bash
python main.py run
```

**Web Interface:**
```bash
streamlit run streamlit_app.py
```

## 🛠️ Key Components

- `src/weave/model.py` - Main RAG model (GPT-4o)
- `src/evaluation/` - Evaluation framework  
- `src/rag/` - Retrieval components
- `src/ocr/` - Document extraction

## 📈 Performance

- **Model**: GPT-4o with 2.3s average latency
- **Retrieval**: 5 documents per query
- **Accuracy**: 97.5% numeric consistency, 85% faithfulness
- **Monitoring**: Full W&B Weave integration