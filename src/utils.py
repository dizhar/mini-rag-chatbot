"""PDF processing utilities for the RAG chatbot."""

import PyPDF2
from typing import List, Dict
import re


def extract_text_from_pdf(pdf_path: str) -> Dict[str, str]:
    """Extract text from PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_by_page = {}
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    text_by_page[f"page_{page_num}"] = clean_text(text)
            
            return text_by_page
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return {}


def clean_text(text: str) -> str:
    """Clean extracted text."""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks


def process_all_pdfs(pdf_paths: List[str]) -> List[Dict]:
    """Process all PDFs and return chunked documents with metadata."""
    all_chunks = []
    
    for pdf_path in pdf_paths:
        # Extract document name from path
        doc_name = pdf_path.split('/')[-1].replace('.pdf', '').replace('_', ' ').title()
        
        # Extract text by page
        pages_text = extract_text_from_pdf(pdf_path)
        
        for page_id, page_text in pages_text.items():
            # Chunk the page text
            chunks = chunk_text(page_text)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'text': chunk,
                    'source': doc_name,
                    'page': page_id.replace('page_', ''),
                    'chunk_id': f"{doc_name}_{page_id}_chunk_{i}"
                })
    
    return all_chunks