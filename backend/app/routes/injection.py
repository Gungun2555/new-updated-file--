# 

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.core.database import get_supabase_client
import tempfile
import docx2txt
import fitz  # PyMuPDF for PDF reading
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import hashlib
import os
from datetime import datetime

router = APIRouter(prefix="/injection", tags=["Injection"])

# Initialize ChromaDB client (Persistent storage)
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

# Initialize embedding model (converts text to vectors)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Get or create collection
collection = chroma_client.get_or_create_collection(
    name="document_store",
    metadata={"hnsw:space": "cosine"}
)


def extract_text_from_file(file_path: str, filename: str) -> str:
    """Extract text from PDF, DOCX, or TXT files"""
    text = ""
    
    if filename.endswith(".pdf"):
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    elif filename.endswith(".docx"):
        text = docx2txt.process(file_path)
    elif filename.endswith(".txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file type. Please upload PDF, DOCX, or TXT files"
        )
    
    return text.strip()


def generate_document_id(uploader_name: str, table_name: str, filename: str, timestamp: str) -> str:
    """Generate unique document ID using MD5 hash"""
    content = f"{uploader_name}_{table_name}_{filename}_{timestamp}"
    return hashlib.md5(content.encode()).hexdigest()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """
    Split text into overlapping chunks for better context preservation
    
    Args:
        text: Full document text
        chunk_size: Characters per chunk (default: 1000)
        overlap: Overlapping characters between chunks (default: 200)
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    
    return chunks


@router.post("/upload")
async def inject_data(
    uploader_name: str = Form(...),
    table_name: str = Form(...),
    comment: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Upload and store document in ChromaDB and Supabase
    
    Process:
    1. Extract text from uploaded file (PDF/DOCX/TXT)
    2. Split text into overlapping chunks
    3. Generate embeddings (vector representations) for each chunk
    4. Store chunks with embeddings in ChromaDB for semantic search
    5. Store document metadata in Supabase for tracking and filtering
    
    Args:
        uploader_name: Name of the person uploading the document
        table_name: Category/table this document belongs to
        comment: Description or notes about the document
        file: The actual file (PDF, DOCX, or TXT)
    
    Returns:
        Success message with document details
    """
    
    supabase = get_supabase_client()
    timestamp = datetime.utcnow().isoformat()
    
    # Step 1: Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Step 2: Extract text from the file
        extracted_text = extract_text_from_file(tmp_path, file.filename)
        
        if not extracted_text:
            raise HTTPException(
                status_code=400, 
                detail="No text could be extracted from the file"
            )
        
        # Step 3: Generate unique document ID
        doc_id = generate_document_id(uploader_name, table_name, file.filename, timestamp)
        
        # Step 4: Split text into chunks (for better retrieval and context)
        text_chunks = chunk_text(extracted_text)
        
        # Step 5: Generate embeddings (convert text to vectors for semantic search)
        chunk_embeddings = embedding_model.encode(text_chunks).tolist()
        
        # Step 6: Prepare data for ChromaDB
        chunk_ids = []
        chunk_metadatas = []
        
        for i, chunk in enumerate(text_chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            chunk_metadatas.append({
                "doc_id": doc_id,
                "uploader_name": uploader_name,
                "table_name": table_name,
                "comment": comment,
                "filename": file.filename,
                "file_type": os.path.splitext(file.filename)[1],
                "timestamp": timestamp,  # Added timestamp in metadata
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "chunk_size": len(chunk)
            })
        
        # Step 7: Store in ChromaDB (vector database for semantic search)
        collection.add(
            ids=chunk_ids,
            embeddings=chunk_embeddings,
            documents=text_chunks,
            metadatas=chunk_metadatas
        )
        
        # Step 8: Store metadata in Supabase (for tracking and structured queries)
        supabase_record = {
            "doc_id": doc_id,
            "uploader_name": uploader_name,
            "table_name": table_name,
            "comment": comment,
            "filename": file.filename,
            "file_type": os.path.splitext(file.filename)[1],
            "file_size_bytes": len(content),
            "extracted_text_length": len(extracted_text),
            "num_chunks": len(text_chunks),
            "timestamp": timestamp,
            "status": "processed"
        }
        
        supabase.table("document_metadata").insert(supabase_record).execute()
        
        return {
            "message": "Document uploaded and processed successfully",
            "doc_id": doc_id,
            "uploader_name": uploader_name,
            "table_name": table_name,
            "filename": file.filename,
            "chunks_created": len(text_chunks),
            "text_length": len(extracted_text),
            "timestamp": timestamp
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing document: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)