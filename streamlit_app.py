import streamlit as st
import os
from pathlib import Path
import tempfile
from PIL import Image

from src.ocr.extractor import OCRExtractor
from src.rag.embed import OpenAIEmbedder
from src.rag.vectore_store import PineconeVectorStore
from src.rag.retriever import Retriever
from src.rag.chunker import chunk_text_by_line, chunk_text_with_overlap, chunk_text_by_paragraph
import weave

# Page config
st.set_page_config(
    page_title="OCR Document Processing",
    page_icon="📄",
    layout="wide"
)

# Initialize components (cached to avoid reloading)
@st.cache_resource
def init_components():
    weave.init("ocr-mrm-streamlit")
    ocr = OCRExtractor()
    embedder = OpenAIEmbedder()
    vector_store = PineconeVectorStore(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name="ocr-mrm-db"
    )
    vector_store.initialize_index(dimension=1536)
    retriever = Retriever(index_name="ocr-mrm-db", namespace="default")
    return ocr, embedder, vector_store, retriever

ocr_extractor, embedder, vector_store, retriever = init_components()

# Title and description
st.title("📄 OCR Document Processing System")
st.markdown("Upload documents for OCR extraction, embedding, and semantic search")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a function", ["Upload & Process", "Search Documents", "View Statistics"])

if page == "Upload & Process":
    st.header("Upload Document for OCR Processing")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
        help="Upload an image containing text for OCR processing"
    )
    
    # Use a fixed namespace for all documents
    namespace = "default"
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Processing Status")
            
            # Process button
            if st.button("Process Document", type="primary"):
                with st.spinner("Processing..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            tmp_path = tmp_file.name
                        
                        # Step 1: OCR Extraction
                        with st.status("Extracting text with OCR...", expanded=True) as status:
                            extracted_text = ocr_extractor.extract_text_from_image_local(tmp_path)
                            if extracted_text:
                                status.update(label="✅ Text extraction complete", state="complete")
                                st.success(f"Extracted {len(extracted_text)} characters")
                            else:
                                status.update(label="❌ Text extraction failed", state="error")
                                st.error("No text extracted")
                                st.stop()
                        
                        # Display extracted text
                        with st.expander("View Extracted Text"):
                            st.text_area("Extracted Text", extracted_text, height=200)

                        # Step 2: Chunk the text
                        with st.status("Chunking text...", expanded=True) as status:
                            # Determine best chunking method based on text characteristics
                            text_length = len(extracted_text)
                            has_paragraphs = '\n\n' in extracted_text
                            
                            if text_length > 1000 and has_paragraphs:
                                # Use paragraph chunking for longer texts with clear structure
                                chunks = chunk_text_by_paragraph(extracted_text, min_length=200)
                                chunk_method = "paragraph"
                            elif text_length > 500:
                                # Use overlapping chunks for medium texts
                                chunk_data = chunk_text_with_overlap(extracted_text, chunk_size=400, overlap=50)
                                chunks = [c['text'] for c in chunk_data]
                                chunk_method = "overlapping"
                            else:
                                # Use line-by-line for short texts
                                chunks = chunk_text_by_line(extracted_text)
                                chunk_method = "line-by-line"
                            
                            if chunks:
                                status.update(label=f"✅ Created {len(chunks)} chunks using {chunk_method} method", state="complete")
                                st.success(f"Created {len(chunks)} chunks using {chunk_method} chunking")
                            else:
                                status.update(label="❌ No chunks created", state="error")
                                st.error("Text could not be split into chunks.")
                                st.stop()

                        with st.expander("View Text Chunks"):
                            st.write(chunks)
                        
                        # Step 3 & 4: Create Embeddings and Store in Pinecone
                        with st.status("Creating embeddings and storing in vector database...", expanded=True) as status:
                            vectors_to_upsert = []
                            base_doc_id = f"{namespace}_{uploaded_file.name}"

                            for i, chunk in enumerate(chunks):
                                # Create embedding for the chunk
                                embedding = embedder.create_embeddings(chunk)
                                if embedding:
                                    # Prepare metadata for the chunk
                                    chunk_metadata = {
                                        "filename": uploaded_file.name,
                                        "text": chunk,
                                        "chunk_number": i + 1,
                                        "document_id": base_doc_id
                                    }
                                    # Create a unique ID for each chunk
                                    chunk_id = f"{base_doc_id}_{i+1}"
                                    
                                    vectors_to_upsert.append((chunk_id, embedding, chunk_metadata))

                            if vectors_to_upsert:
                                vector_store.upsert(
                                    vectors=vectors_to_upsert,
                                    namespace=namespace
                                )
                                status.update(label=f"✅ Stored {len(vectors_to_upsert)} chunks in database", state="complete")
                            else:
                                status.update(label="❌ No vectors to store", state="error")
                                st.stop()
                        
                        # Success message
                        st.balloons()
                        st.success(f"Document '{uploaded_file.name}' processed and chunked successfully!")
                        
                        # Clean up
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")

elif page == "Search Documents":
    st.header("🔍 Search Documents")
    
    # Search interface
    query = st.text_input("Enter your search query", placeholder="What are you looking for?")
    
    # Use a fixed namespace and adjust layout
    namespace = "default"
    col1, col2 = st.columns([3, 1])
    with col1:
        top_k = st.slider("Number of results", 1, 10, 5)
    with col2:
        use_reranking = st.checkbox("Use reranking", value=True)
    
    if st.button("Search", type="primary") and query:
        with st.spinner("Searching..."):
            try:
                if use_reranking:
                    results = retriever.retrieve_and_rerank(query, top_k=top_k, namespace=namespace)
                else:
                    results = retriever.retrieve(query, top_k=top_k, namespace=namespace)
                
                if results:
                    st.success(f"Found {len(results)} results")
                    
                    # Display results
                    for i, result in enumerate(results):
                        metadata = result.get("metadata", {})
                        score = result.get("score", 0)
                        
                        with st.expander(f"Result {i+1}: {metadata.get('filename', 'Unknown')} (Score: {score:.4f})"):
                            st.write(f"**Chunk {metadata.get('chunk_number', 0)}:**")
                            st.info(metadata.get('text', ''))
                else:
                    st.warning("No results found")
                    
            except Exception as e:
                st.error(f"Search error: {str(e)}")

elif page == "View Statistics":
    st.header("📊 Index Statistics")
    
    try:
        stats = vector_store.describe_index_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Vectors", stats.get("total_vector_count", 0))
        with col2:
            st.metric("Dimension", stats.get("dimension", 0))
        
        st.subheader("Namespace Information")
        namespaces = stats.get("namespaces", {})
        if namespaces:
            for ns, count in namespaces.items():
                st.write(f"**{ns}**: {count} vectors")
        else:
            st.info("No namespaces found")
            
    except Exception as e:
        st.error(f"Error fetching statistics: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This application uses OCR to extract text from images, "
    "creates embeddings, and stores them in Pinecone for "
    "semantic search capabilities."
)