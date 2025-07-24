import streamlit as st
import os
from pathlib import Path
import tempfile
from PIL import Image
import weave

from src.weave.model import RagModel

# --- Page Configuration ---
st.set_page_config(
    page_title="RAG Document Processing with Weave",
    page_icon="🚀",
    layout="wide"
)

# --- Weave & Model Initialization ---
@st.cache_resource
def init_model():
    """Initializes the Weave project and the RAG model."""
    weave.init("ocr-mrm-streamlit-v2")
    
    # Configuration for our model.
    # We can centralize and easily change this.
    model = RagModel(
        index_name="ocr-mrm-db",
        namespace="default"
    )
    return model

model = init_model()

# --- UI Layout ---
st.title("🚀 RAG Document Processing with Weave")
st.markdown("Upload documents for OCR, processing, and semantic search, all versioned by Weave.")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a function", ["Upload & Process", "Search Documents", "View Statistics"])

# --- Page 1: Upload & Process ---
if page == "Upload & Process":
    st.header("Upload Document for Processing")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
        help="The document will be processed and stored in the vector database."
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Processing")
            if st.button("Process Document", type="primary"):
                with st.spinner("Processing document with the RAG model..."):
                    try:
                        # Use a temporary file to pass the path to the model
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            tmp_path = tmp_file.name
                        
                        # Call the model's processing method
                        # This is a tracked Weave operation
                        result = model.load_and_process_document(tmp_path)
                        
                        # Clean up the temporary file
                        os.unlink(tmp_path)

                        st.balloons()
                        st.success(f"Document processed successfully! {result['processed_chunks']} chunks were created and stored.")
                        
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

# --- Page 2: Search Documents ---
elif page == "Search Documents":
    st.header("🔍 Search Processed Documents")
    
    query = st.text_input("Enter your search query", placeholder="e.g., 'What is the total amount?'")
    
    top_k = st.slider("Number of results to retrieve", 1, 10, 5)
    
    if st.button("Search", type="primary") and query:
        with st.spinner("Searching for relevant documents and generating an answer..."):
            try:
                # Call the model's predict method
                # This is a tracked Weave operation
                prediction = model.predict(query, top_k=top_k)
                
                # Display the generated answer
                generated_answer = prediction.get("generated_answer")
                if generated_answer:
                    st.subheader("🤖 Generated Answer")
                    st.markdown(generated_answer)
                else:
                    st.warning("The model did not generate an answer.")

                # Display the retrieved chunks
                results = prediction.get("raw_matches", [])
                
                if results:
                    st.success(f"Found {len(results)} relevant chunks.")
                    for i, res in enumerate(results):
                        metadata = res.get("metadata", {})
                        score = res.get("score", 0)
                        
                        with st.expander(f"Result {i+1} (Score: {score:.4f}) - {metadata.get('filename', 'N/A')}"):
                            st.info(metadata.get('text', 'No text available.'))
                else:
                    st.warning("No results found for your query.")
                    
            except Exception as e:
                st.error(f"An error occurred during search: {e}")

# --- Page 3: View Statistics ---
elif page == "View Statistics":
    st.header("📊 Vector Store Statistics")
    
    if st.button("Refresh Statistics"):
        with st.spinner("Fetching index statistics..."):
            try:
                # Call the model's method to get stats
                # This is a tracked Weave operation
                stats = model.get_index_stats()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Vectors", stats.get("total_vector_count", 0))
                with col2:
                    st.metric("Index Dimension", stats.get("dimension", 0))
                
                st.subheader("Vectors per Namespace")
                namespaces = stats.get("namespaces", {})
                if namespaces:
                    st.table(namespaces.items())
                else:
                    st.info("No namespaces found in the index.")
                    
            except Exception as e:
                st.error(f"Failed to fetch statistics: {e}")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This app demonstrates a RAG system powered by Weave, "
    "allowing for versioned, trackable document processing and retrieval."
)