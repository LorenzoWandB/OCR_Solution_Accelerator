import streamlit as st
import os
from pathlib import Path
import tempfile
import json
from PIL import Image
import weave
import asyncio
from datetime import datetime

from src.weave.model import RagModel
from src.evaluation.weave_native_eval import run_weave_native_evaluation_sync
from src.evaluation.dataset_creator import create_synthetic_evaluation_dataset
from src.llamaindex.extractor import extract_documents


# --- Page Configuration ---
st.set_page_config(
    page_title="RAG Playground - LlamaIndex + Pinecone",
    page_icon="🎯",
    layout="wide"
)

# --- Custom CSS for better UI ---
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #1f2937;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .stage-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stage-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .stage-subtitle {
        font-size: 0.875rem;
        color: #6b7280;
        margin-bottom: 1rem;
    }
    .waiting-text {
        color: #9ca3af;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# --- Weave & Model Initialization ---
@st.cache_resource
def init_model():
    """Initializes the Weave project and the RAG model."""
    weave.init("solution-accelerator-mrm-eval")
    
    model = RagModel(
        index_name="ocr-mrm-db",
        namespace="default"
    )
    return model

model = init_model()

# --- Session State Initialization ---
if 'extraction_result' not in st.session_state:
    st.session_state.extraction_result = None
if 'reasoning_result' not in st.session_state:
    st.session_state.reasoning_result = None
if 'embeddings_result' not in st.session_state:
    st.session_state.embeddings_result = None
if 'retrieval_result' not in st.session_state:
    st.session_state.retrieval_result = None
if 'query_running' not in st.session_state:
    st.session_state.query_running = False
if 'latest_extraction' not in st.session_state:
    st.session_state.latest_extraction = None

# --- Header ---
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<p class="main-header">🎯 RAG Playground</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">LlamaIndex • Pinecone</p>', unsafe_allow_html=True)
with col2:
    st.markdown("**Extraction • Reasoning • Embeddings • Retrieval**")

# --- Main Tabs ---
tab1, tab2, tab3 = st.tabs(["🎮 RAG Playground", "📤 Upload Documents", "📊 Run Evaluation"])

# ============================================================================
# TAB 1: RAG PLAYGROUND
# ============================================================================
with tab1:
    # Two-column layout
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        st.markdown("### Ask your data")
        st.caption("Enter a question. We'll fetch vectors (Pinecone), run LlamaIndex, and show everything.")
        
        # Query input
        query = st.text_area(
            "Query",
            placeholder="e.g., Extract key dates and decisions from the design RFC",
            height=150,
            label_visibility="collapsed"
        )
        
        # Run button
        col_btn1, col_btn2 = st.columns([6, 1])
        with col_btn1:
            run_button = st.button("▶ Run", type="primary", use_container_width=True)
        with col_btn2:
            clear_button = st.button("🗑️", use_container_width=True)
        
        if clear_button:
            st.session_state.extraction_result = None
            st.session_state.reasoning_result = None
            st.session_state.embeddings_result = None
            st.session_state.retrieval_result = None
            st.rerun()
        
        # Results section
        st.markdown("---")
        
        if run_button and query:
            st.session_state.query_running = True
            
            with st.spinner("🔍 Processing your query..."):
                try:
                    # Run the full RAG pipeline
                    prediction = model.predict(query, top_k=5)
                    
                    # Store results in session state
                    st.session_state.retrieval_result = prediction.get("raw_matches", [])
                    st.session_state.reasoning_result = prediction.get("generated_answer", "")
                    st.session_state.extraction_result = {
                        "query": query,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.embeddings_result = {
                        "chunks_retrieved": len(prediction.get("raw_matches", []))
                    }
                    
                    st.session_state.query_running = False
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.exception(e)
                    st.session_state.query_running = False
        
        # Display results
        if st.session_state.reasoning_result:
            st.markdown("### 🤖 Generated Answer")
            st.markdown(st.session_state.reasoning_result)
            
            if st.session_state.retrieval_result:
                st.markdown("### 📚 Retrieved Chunks")
                for i, result in enumerate(st.session_state.retrieval_result):
                    metadata = result.get("metadata", {})
                    score = result.get("score", 0)
                    
                    with st.expander(f"Chunk {i+1} • Score: {score:.4f} • {metadata.get('filename', 'N/A')}"):
                        st.markdown(metadata.get('text', 'No text available'))
        else:
            st.info("Results will appear here.")
    
    with right_col:
        # ---- Extraction Stage ----
        st.markdown('<div class="stage-card">', unsafe_allow_html=True)
        st.markdown('<p class="stage-title">Extraction</p>', unsafe_allow_html=True)
        st.markdown('<p class="stage-subtitle">LlamaIndex structured extraction</p>', unsafe_allow_html=True)
        
        if st.session_state.latest_extraction:
            st.success("✓ Document Extracted")
            # Showing the first few fields from the latest extraction
            if st.session_state.latest_extraction[0]['data']:
                first_key = list(st.session_state.latest_extraction[0]['data'].keys())[0]
                first_val = st.session_state.latest_extraction[0]['data'][first_key]
                st.caption(f"e.g., {first_key}: {first_val}")
        elif st.session_state.extraction_result:
            st.success("✓ Query processed")
            st.caption(f"Timestamp: {st.session_state.extraction_result.get('timestamp', 'N/A')}")
        else:
            st.markdown('<p class="waiting-text">Waiting for a run...</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ---- Reasoning Stage ----
        st.markdown('<div class="stage-card">', unsafe_allow_html=True)
        st.markdown('<p class="stage-title">Reasoning</p>', unsafe_allow_html=True)
        st.markdown('<p class="stage-subtitle">Planner & Tool Calls (from API)</p>', unsafe_allow_html=True)
        
        if st.session_state.reasoning_result:
            st.success("✓ Answer generated")
            st.caption(f"Length: {len(st.session_state.reasoning_result)} chars")
        else:
            st.markdown('<p class="waiting-text">Waiting for a run...</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ---- Embeddings Stage ----
        st.markdown('<div class="stage-card">', unsafe_allow_html=True)
        st.markdown('<p class="stage-title">Embedded to Pinecone</p>', unsafe_allow_html=True)
        st.markdown('<p class="stage-subtitle">RAG vectors that were upserted</p>', unsafe_allow_html=True)
        
        if st.session_state.embeddings_result:
            st.success("✓ Vectors retrieved")
            st.caption(f"Chunks: {st.session_state.embeddings_result.get('chunks_retrieved', 0)}")
        else:
            st.markdown('<p class="waiting-text">No embeddings reported yet.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ---- Retrieval Stage ----
        st.markdown('<div class="stage-card">', unsafe_allow_html=True)
        st.markdown('<p class="stage-title">Pinecone Retrieval</p>', unsafe_allow_html=True)
        st.markdown('<p class="stage-subtitle">Top matches for the user query</p>', unsafe_allow_html=True)
        
        if st.session_state.retrieval_result:
            st.success(f"✓ {len(st.session_state.retrieval_result)} matches found")
            if st.session_state.retrieval_result:
                top_score = st.session_state.retrieval_result[0].get('score', 0)
                st.caption(f"Top score: {top_score:.4f}")
        else:
            st.markdown('<p class="waiting-text">No matches yet. Run a query.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 2: UPLOAD DOCUMENTS
# ============================================================================
with tab2:
    st.markdown("### 📤 Upload Document for Processing")
    st.markdown("Use LlamaIndex to extract structured information from financial documents, then chunk, embed, and store in Pinecone.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Document Upload")
        
        uploaded_file = st.file_uploader(
            "Upload an income statement for extraction",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            help="The document will be processed by LlamaExtract to extract structured data."
        )

    with col2:
        st.markdown("#### Processing Pipeline")
        st.info(
            "**Steps:**\n"
            "1. 📄 Extract with LlamaIndex\n"
            "2. 🤔 Extracted Data Below\n"
            "3. ⏭️ Next: Chunk, Embed, Store\n"
        )
    
    if uploaded_file is not None:
        st.markdown("---")
        
        col_preview, col_process = st.columns([1, 1])
        
        with col_preview:
            st.markdown("#### 📎 File Preview")
            if uploaded_file.type.startswith('image'):
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
            else:
                st.info(f"**File**: {uploaded_file.name}")
                st.info(f"**Type**: {uploaded_file.type}")
                st.info(f"**Size**: {uploaded_file.size / 1024:.2f} KB")
        
        with col_process:
            st.markdown("#### ⚙️ Process Document")
            
            if st.button("🚀 Extract Data", type="primary", use_container_width=True):
                with st.spinner("Extracting data from document... This may take a moment."):
                    try:
                        # Call our new async extractor directly with the uploaded file object
                        # The [uploaded_file] is because the function expects an iterable
                        results = asyncio.run(extract_documents([uploaded_file]))
                        
                        # Store the latest extraction result in the session state
                        st.session_state.latest_extraction = results
                        
                        st.balloons()
                        st.success("✅ Extraction successful!")
                        
                    except Exception as e:
                        st.error(f"❌ An error occurred during extraction: {e}")
                        st.exception(e)
                        st.info("""
                        **Troubleshooting:**
                        - Check your API keys (OPENAI_API_KEY, LLAMA_CLOUD_API_KEY)
                        - Ensure the document is a clear, readable income statement
                        """)

        # Display the extraction results if they exist
        if st.session_state.latest_extraction:
            st.markdown("---")
            st.markdown("###  extracted data")
            st.caption("This is the structured data extracted from your document, ready to be sent to Pinecone.")
            
            # We only process one file at a time in this UI, so we show the first result
            extraction_output = st.session_state.latest_extraction[0]
            st.json(extraction_output)


# ============================================================================
# TAB 3: RUN EVALUATION
# ============================================================================
with tab3:
    st.markdown("### 📊 Run RAG Evaluation")
    st.markdown("Comprehensive evaluation of your RAG pipeline with multiple quality metrics.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### ⚙️ Evaluation Settings")
        
        eval_mode = st.radio(
            "Dataset Source",
            ["Generate Synthetic Dataset", "Use Existing Dataset"],
            help="Choose to generate synthetic test data or use a pre-existing dataset"
        )
        
        if eval_mode == "Generate Synthetic Dataset":
            eval_dataset_size = st.number_input(
                "Number of test examples",
                min_value=5,
                max_value=100,
                value=20,
                help="Number of Q&A pairs to generate and evaluate"
            )
            st.info("Will generate synthetic financial Q&A pairs for evaluation")
        else:
            st.warning("⚠️ Custom dataset loading not yet implemented. Please use synthetic generation.")
    
    with col2:
        st.markdown("#### 🎯 Metrics to Track")
        st.markdown("""
        - ✅ **Recall@K**: Retrieval accuracy
        - ✅ **MRR**: Mean Reciprocal Rank
        - ✅ **Faithfulness**: Answer grounding
        - ✅ **Numeric Consistency**: Financial data accuracy
        - ✅ **Latency**: Response time (via Weave)
        - ✅ **Cost**: Token usage (via Weave)
        """)
    
    st.markdown("---")
    
    # Run evaluation button
    if st.button("🚀 Run Evaluation", type="primary", use_container_width=False):
        if eval_mode == "Use Existing Dataset":
            st.error("Please select 'Generate Synthetic Dataset' for now.")
        else:
            with st.spinner("Running evaluation... This may take several minutes."):
                try:
                    # Initialize progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: Create dataset
                    status_text.text("📝 Step 1/3: Creating synthetic evaluation dataset...")
                    progress_bar.progress(10)
                    
                    dataset_info = create_synthetic_evaluation_dataset(
                        index_name="ocr-mrm-db",
                        namespace="default"
                    )
                    dataset_ref = dataset_info['dataset_object']
                    
                    progress_bar.progress(40)
                    status_text.text(f"✓ Created {dataset_info['total_qa_pairs']} test examples")
                    
                    # Step 2: Run evaluation
                    status_text.text("📝 Step 2/3: Running RAG evaluation...")
                    progress_bar.progress(50)
                    
                    evaluation_result = run_weave_native_evaluation_sync(
                        index_name="ocr-mrm-db",
                        namespace="default",
                        dataset_ref=dataset_ref,
                        k=5
                    )
                    
                    progress_bar.progress(90)
                    status_text.text("📝 Step 3/3: Finalizing results...")
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Evaluation complete!")
                    
                    # Display results
                    st.balloons()
                    st.success("🎉 Evaluation completed successfully!")
                    
                    st.markdown("---")
                    st.markdown("### 📈 Evaluation Results")
                    
                    # Extract evaluation results
                    eval_results = evaluation_result.get('evaluation_results', {})
                    model_config = evaluation_result.get('model_config', {})
                    
                    # Show configuration
                    config_col1, config_col2, config_col3, config_col4 = st.columns(4)
                    with config_col1:
                        st.metric("Test Examples", dataset_info['total_qa_pairs'])
                    with config_col2:
                        st.metric("Top-K", model_config.get('k', 5))
                    with config_col3:
                        st.metric("Index", model_config.get('index_name', 'N/A'))
                    with config_col4:
                        st.metric("Namespace", model_config.get('namespace', 'N/A'))
                    
                    # Show detailed results
                    st.markdown("---")
                    with st.expander("📊 Detailed Evaluation Results", expanded=True):
                        st.json(evaluation_result)
                    
                    # Weave dashboard link
                    st.info("🔗 View detailed traces, metrics, and comparisons in your [Weave dashboard](https://wandb.ai/)")
                    
                    # Helpful tips
                    with st.expander("💡 Next Steps"):
                        st.markdown("""
                        **Analyzing Your Results:**
                        1. Check the **Weave dashboard** for detailed traces of each evaluation
                        2. Look at **Recall@K** to understand retrieval quality
                        3. Review **Faithfulness** scores to ensure answers are grounded in context
                        4. Monitor **Numeric Consistency** for financial accuracy
                        5. Compare multiple runs to track improvements
                        
                        **Improving Performance:**
                        - Adjust chunk size and overlap in model configuration
                        - Experiment with different embedding models
                        - Fine-tune retrieval top-K parameter
                        - Add more diverse training documents
                        """)
                    
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {e}")
                    st.exception(e)
    
    # Evaluation history section
    st.markdown("---")
    st.markdown("### 📜 Evaluation History")
    st.info("💡 View all past evaluations and compare performance over time in your Weave dashboard at https://wandb.ai/")
    
    if st.button("Open Weave Dashboard 🔗"):
        st.markdown("[🔗 Open Weave Dashboard](https://wandb.ai/)", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.markdown("### 🎯 RAG Playground")
st.sidebar.markdown("**Built for rapid iteration**")
st.sidebar.caption("Replace the fetch() with your backend that uses LlamaIndex + Pinecone.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Current Configuration")
st.sidebar.text(f"Index: {model.index_name}")
st.sidebar.text(f"Namespace: {model.namespace}")
st.sidebar.text(f"Embedding: {model.embedding_model}")
st.sidebar.text(f"LLM: {model.llm_model}")
st.sidebar.text(f"Top-K: {model.retriever_top_k}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Tech Stack")
st.sidebar.info(
    "**Powered by:**\n"
    "- 🔍 LlamaIndex\n"
    "- 🎯 Pinecone\n"
    "- 📊 Weave (W&B)\n"
    "- 🤖 OpenAI\n"
    "- 🎨 Streamlit"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Resources")
st.sidebar.markdown("""
- [Weave Dashboard](https://wandb.ai/)
- [LlamaIndex Docs](https://docs.llamaindex.ai/)
- [Pinecone Docs](https://docs.pinecone.io/)
""")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("🎮 RAG Playground v2.0")
