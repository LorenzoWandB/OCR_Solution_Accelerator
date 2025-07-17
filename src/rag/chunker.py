import weave
from typing import List, Dict, Any

@weave.op()
def chunk_text_by_line(text: str) -> list[str]:
    """
    Splits the text into chunks by newline characters.

    Args:
        text: The input text to be chunked.

    Returns:
        A list of text chunks.
    """
    if not isinstance(text, str):
        return []
    
    chunks = [chunk.strip() for chunk in text.split('\n') if chunk.strip()]
    return chunks

@weave.op()
def chunk_text_with_overlap(text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Splits text into overlapping chunks for better context preservation.
    
    Args:
        text: The input text to be chunked
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of dictionaries containing chunk text and metadata
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    chunks = []
    text = text.strip()
    start = 0
    chunk_num = 0
    
    while start < len(text):
        # Calculate end position
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence ending punctuation
            for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                last_punct = text.rfind(punct, start, end)
                if last_punct != -1:
                    end = last_punct + 1
                    break
        
        # Extract chunk
        chunk_text = text[start:end].strip()
        
        if chunk_text:
            chunks.append({
                'text': chunk_text,
                'chunk_num': chunk_num,
                'start_char': start,
                'end_char': min(end, len(text))
            })
            chunk_num += 1
        
        # Move start position
        start = end - overlap if end < len(text) else end
    
    return chunks

@weave.op()
def chunk_text_by_paragraph(text: str, min_length: int = 100) -> List[str]:
    """
    Splits text into chunks by paragraphs (double newlines).
    Combines short paragraphs to meet minimum length.
    
    Args:
        text: The input text to be chunked
        min_length: Minimum length for a chunk
        
    Returns:
        List of text chunks
    """
    if not isinstance(text, str):
        return []
    
    # Split by double newlines (paragraphs)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_length = len(para)
        
        if current_length + para_length < min_length:
            # Add to current chunk
            current_chunk.append(para)
            current_length += para_length
        else:
            # Save current chunk if it exists
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            # Start new chunk
            current_chunk = [para]
            current_length = para_length
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks 