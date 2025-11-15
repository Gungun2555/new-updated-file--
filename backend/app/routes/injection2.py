"""
Smart Document Injection: Upload → Extract → Generate SQL → Execute → Store with Audit Trail
Processes documents to automatically update target_list with full history tracking in Supabase and Pinecone
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.core.database import get_supabase_client
from app.core.pinecone_client import pc
import tempfile
import docx2txt
import fitz  # PyMuPDF for PDF reading
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import hashlib
import os
import psycopg2
import psycopg2.extras
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
import json
import re

load_dotenv()

router = APIRouter(prefix="/injection", tags=["Injection"])

# Initialize services
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemini-2.0-flash")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Get Pinecone index
index_name = os.getenv("PINECONE_INDEX_NAME", "document-store")
index = pc.Index(index_name)

# Database connection
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

# Target List Schema for LLM context
TARGET_LIST_SCHEMA = """
CREATE TABLE target_list (
  id SERIAL PRIMARY KEY,
  hcp_code TEXT UNIQUE,
  full_name TEXT NOT NULL,
  gender TEXT CHECK (gender IN ('male', 'female', 'other')),
  qualification TEXT,
  specialty TEXT,
  designation TEXT,
  email TEXT,
  phone TEXT,
  hospital_name TEXT,
  hospital_address TEXT,
  city TEXT,
  state TEXT,
  pincode TEXT,
  experience_years INTEGER,
  influence_score NUMERIC(5,2),
  category TEXT,
  therapy_area TEXT,
  monthly_sales INTEGER,
  yearly_sales INTEGER,
  last_interaction_date DATE,
  call_frequency INTEGER,
  priority BOOLEAN DEFAULT false
);

Changes are automatically logged to history_table via triggers.
"""


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


def generate_document_id(uploader_name: str, filename: str, timestamp: str) -> str:
    """Generate unique document ID using MD5 hash"""
    content = f"{uploader_name}_{filename}_{timestamp}"
    return hashlib.md5(content.encode()).hexdigest()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
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


def extract_entities_with_llm(document_text: str) -> dict:
    """
    Use LLM to extract structured data from document
    Returns: {action, hcp_data, reason, identifier}
    """
    prompt = f"""
    Analyze this document and extract healthcare professional information.
    
    DOCUMENT:
    {document_text[:3000]}
    
    Extract and return JSON with this structure:
    {{
        "action": "INSERT|UPDATE|DELETE",
        "hcp_data": {{
            "hcp_code": "code if mentioned",
            "full_name": "full name",
            "gender": "male|female|other",
            "qualification": "MBBS, MD, etc",
            "specialty": "Cardiology, etc",
            "designation": "role",
            "email": "email@example.com",
            "phone": "phone number",
            "hospital_name": "hospital name",
            "hospital_address": "address",
            "city": "city",
            "state": "state",
            "pincode": "pincode",
            "experience_years": 0,
            "influence_score": 0.0,
            "category": "A|B|C|D",
            "therapy_area": "therapy area",
            "monthly_sales": 0,
            "yearly_sales": 0,
            "last_interaction_date": "YYYY-MM-DD",
            "call_frequency": 0,
            "priority": true|false
        }},
        "reason": "reason for this change",
        "identifier": "email or full_name to find existing record for UPDATE/DELETE"
    }}
    
    Rules:
    - Return ONLY valid JSON, no markdown
    - Extract numeric values as numbers, not strings
    - If value not mentioned, use null
    - Determine action: INSERT (new HCP), UPDATE (modify existing), DELETE (remove)
    - For UPDATE/DELETE: provide identifier (email or full_name)
    - For DELETE: explain reason in "reason" field
    """
    
    try:
        response = llm.generate_content(prompt)
        clean_text = response.text.strip()
        clean_text = re.sub(r'```json|```', '', clean_text).strip()
        result = json.loads(clean_text)
        return result
    except Exception as e:
        print(f"❌ Entity extraction error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to extract data: {str(e)}")


def generate_sql_from_document(extracted_data: dict) -> str:
    """
    Generate SQL query based on extracted data and target_list schema
    Handles INSERT, UPDATE, DELETE operations
    """
    prompt = f"""
    Generate a PostgreSQL query to process this data change.
    
    SCHEMA:
    {TARGET_LIST_SCHEMA}
    
    ACTION: {extracted_data['action']}
    DATA: {json.dumps(extracted_data['hcp_data'], indent=2)}
    IDENTIFIER: {extracted_data.get('identifier', '')}
    
    Requirements:
    - Return ONLY the SQL query, no explanation or markdown
    - Use %s for parameters (prepared statement format)
    - For INSERT: Include all non-null fields
    - For UPDATE: Use identifier in WHERE clause (email or full_name)
    - For DELETE: Use identifier in WHERE clause
    - Handle NULL values properly
    
    Examples:
    INSERT INTO target_list (hcp_code, full_name, email, specialty) VALUES (%s, %s, %s, %s)
    UPDATE target_list SET specialty=%s, city=%s WHERE email=%s
    DELETE FROM target_list WHERE email=%s
    """
    
    try:
        response = llm.generate_content(prompt)
        sql = response.text.strip()
        sql = re.sub(r'```sql|```', '', sql).strip()
        
        # Validate SQL
        if not any(keyword in sql.upper() for keyword in ['INSERT', 'UPDATE', 'DELETE']):
            raise ValueError("Invalid SQL: missing INSERT/UPDATE/DELETE")
        
        print(f"📝 Generated SQL:\n{sql}\n")
        return sql
    
    except Exception as e:
        print(f"❌ SQL generation error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to generate SQL: {str(e)}")


def execute_sql_and_get_changes(sql: str, extracted_data: dict) -> tuple:
    """
    Execute SQL query and return (success, changed_rows, change_description)
    Triggers will automatically log to history_table
    """
    if not SUPABASE_DB_URL:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    conn = None
    try:
        conn = psycopg2.connect(SUPABASE_DB_URL)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            hcp_data = extracted_data['hcp_data']
            action = extracted_data['action']
            identifier = extracted_data.get('identifier', '')
            
            # Build parameters based on action
            params = []
            
            if action == 'INSERT':
                # Extract field names from the SQL INSERT statement
                # Parse: INSERT INTO target_list (field1, field2, ...) VALUES
                import re
                match = re.search(r'\((.*?)\)\s+VALUES', sql, re.IGNORECASE)
                if match:
                    fields_str = match.group(1)
                    fields = [f.strip() for f in fields_str.split(',')]
                    
                    # Build params in the same order as SQL fields
                    for field in fields:
                        params.append(hcp_data.get(field))
                    
                    print(f"   📍 Parsed {len(fields)} fields from SQL: {fields}")
                else:
                    # Fallback: use all fields (old behavior)
                    for key in ['hcp_code', 'full_name', 'gender', 'qualification', 'specialty', 
                               'designation', 'email', 'phone', 'hospital_name', 'hospital_address',
                               'city', 'state', 'pincode', 'experience_years', 'influence_score',
                               'category', 'therapy_area', 'monthly_sales', 'yearly_sales',
                               'last_interaction_date', 'call_frequency', 'priority']:
                        params.append(hcp_data.get(key))
            
            elif action == 'UPDATE':
                # Extract SET fields and WHERE field from SQL
                import re
                set_match = re.search(r'SET\s+(.*?)\s+WHERE', sql, re.IGNORECASE)
                if set_match:
                    set_clause = set_match.group(1)
                    # Extract field names (field=%s, field=%s, ...)
                    set_fields = [f.split('=')[0].strip() for f in set_clause.split(',')]
                    
                    # Build params for SET clause
                    for field in set_fields:
                        params.append(hcp_data.get(field))
                    
                    # Add identifier for WHERE clause
                    params.append(identifier)
                    print(f"   📍 Parsed {len(set_fields)} update fields: {set_fields}")
                else:
                    # Fallback
                    for key, value in hcp_data.items():
                        if value is not None:
                            params.append(value)
                    params.append(identifier)
            
            elif action == 'DELETE':
                # Only identifier for WHERE
                params = [identifier]
            
            # Count placeholders in SQL
            placeholder_count = sql.count('%s')
            
            print(f"   📍 SQL has {placeholder_count} placeholders, providing {len(params)} parameters")
            
            if placeholder_count != len(params):
                print(f"   ⚠️  Parameter count mismatch! Adjusting...")
                # Trim or pad params to match
                if len(params) > placeholder_count:
                    params = params[:placeholder_count]
                elif len(params) < placeholder_count:
                    params.extend([None] * (placeholder_count - len(params)))
            
            # Execute query
            cur.execute(sql, params)
            affected_rows = cur.rowcount
            
            conn.commit()
            
            # Build change description
            if action == 'DELETE':
                change_desc = f"Removed {affected_rows} HCP. Reason: {extracted_data.get('reason', 'Not specified')}"
            elif action == 'UPDATE':
                change_desc = f"Updated {affected_rows} HCP record(s)"
            else:  # INSERT
                change_desc = f"Added {affected_rows} new HCP"
            
            return True, affected_rows, change_desc
    
    except Exception as e:
        print(f"❌ SQL execution error: {e}")
        if conn:
            conn.rollback()
        raise HTTPException(status_code=400, detail=f"Database error: {str(e)}")
    
    finally:
        if conn:
            conn.close()


@router.post("/upload")
async def inject_data(
    uploader_name: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Smart Document Upload: Extract → Generate SQL → Execute → Store
    
    Process Flow:
    1. Extract text from document (PDF/DOCX/TXT)
    2. Use LLM to identify action (INSERT/UPDATE/DELETE) and extract HCP data
    3. Generate SQL query based on target_list schema
    4. Execute SQL on target_list (triggers auto-update history_table)
    5. Store document chunks in Pinecone with doc_id
    6. Link doc_id in history_table for full traceability
    
    Args:
        uploader_name: Name of person uploading
        file: Document file (PDF, DOCX, or TXT)
    
    Returns:
        Success message with doc_id, action taken, and changes made
    """
    
    supabase = get_supabase_client()
    timestamp = datetime.utcnow().isoformat()
    table_name = "target_list"
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        print(f"\n{'='*70}")
        print(f"🚀 SMART DOCUMENT INJECTION STARTED")
        print(f"{'='*70}")
        
        # Step 1: Extract text from document
        print(f"1️⃣  Extracting text from {file.filename}...")
        extracted_text = extract_text_from_file(tmp_path, file.filename)
        
        if not extracted_text:
            raise HTTPException(status_code=400, detail="No text could be extracted from the file")
        
        print(f"   ✅ Extracted {len(extracted_text)} characters")
        
        # Step 2: Use LLM to extract structured data
        print(f"2️⃣  Analyzing document with AI...")
        extracted_data = extract_entities_with_llm(extracted_text)
        print(f"   Action: {extracted_data['action']}")
        print(f"   Subject: {extracted_data['hcp_data'].get('full_name', 'Unknown')}")
        
        # Step 3: Generate SQL query
        print(f"3️⃣  Generating SQL query...")
        sql = generate_sql_from_document(extracted_data)
        
        # Step 4: Execute SQL on target_list
        print(f"4️⃣  Executing query on target_list...")
        success, affected_rows, change_desc = execute_sql_and_get_changes(sql, extracted_data)
        print(f"   ✅ {change_desc}")
        
        # Step 5: Generate doc_id and prepare for storage
        print(f"5️⃣  Preparing document for storage...")
        doc_id = generate_document_id(uploader_name, file.filename, timestamp)
        
        # Split text into chunks
        text_chunks = chunk_text(extracted_text)
        
        # Generate embeddings for semantic search
        chunk_embeddings = embedding_model.encode(text_chunks).tolist()
        
        # Step 6: Store in Pinecone with metadata
        print(f"6️⃣  Storing in Pinecone...")
        vectors_to_upsert = []
        
        for i, (chunk, embedding) in enumerate(zip(text_chunks, chunk_embeddings)):
            chunk_id = f"{doc_id}_chunk_{i}"
            
            metadata = {
                "doc_id": doc_id,
                "uploader_name": uploader_name,
                "table_name": table_name,
                "action": extracted_data['action'],
                "hcp_name": extracted_data['hcp_data'].get('full_name', ''),
                "hcp_email": extracted_data['hcp_data'].get('email', ''),
                "filename": file.filename,
                "file_type": os.path.splitext(file.filename)[1],
                "timestamp": timestamp,
                "chunk_index": str(i),
                "total_chunks": str(len(text_chunks)),
                "chunk_size": str(len(chunk)),
                "chunk_text": chunk[:500],  # First 500 chars for preview
                "change_description": change_desc
            }
            
            vectors_to_upsert.append((chunk_id, embedding, metadata))
        
        # Upload to Pinecone
        index.upsert(vectors=vectors_to_upsert)
        print(f"   ✅ Stored {len(vectors_to_upsert)} chunks in Pinecone")
        
        # Step 7: Update history_table with doc_id
        print(f"7️⃣  Logging to history_table...")
        
        # Get latest version number
        version_resp = supabase.table("history_table").select("version_number") \
            .eq("table_name", table_name) \
            .order("version_number", desc=True) \
            .limit(1) \
            .execute()
        
        last_version = version_resp.data[0]["version_number"] if version_resp.data else 0
        new_version = last_version + 1
        
        # Map action to operation_type
        operation_type_map = {
            'INSERT': 'INSERT',
            'UPDATE': 'UPDATE',
            'DELETE': 'DELETE',
            'REMOVE': 'DELETE'
        }
        
        history_record = {
            "version_number": new_version,
            "operation_type": operation_type_map.get(extracted_data['action'], 'UPDATE'),
            "table_name": table_name,
            "total_rows": None,
            "changed_rows": affected_rows,
            "reason": extracted_data.get('reason', change_desc),
            "triggered_by": uploader_name,
            "timestamp": timestamp,
            "doc_id": doc_id,  # Same doc_id as Pinecone
            "filename": file.filename,
            "file_type": os.path.splitext(file.filename)[1],
            "num_chunks": len(text_chunks)
        }
        
        supabase.table("history_table").insert(history_record).execute()
        print(f"   ✅ Logged to history_table (version {new_version}) with doc_id: {doc_id}")
        
        print(f"\n{'='*70}")
        print(f"✅ DOCUMENT PROCESSING COMPLETE")
        print(f"{'='*70}\n")
        
        return {
            "message": "Document uploaded and processed successfully",
            "doc_id": doc_id,
            "uploader_name": uploader_name,
            "filename": file.filename,
            "action": extracted_data['action'],
            "subject": extracted_data['hcp_data'].get('full_name', 'Unknown'),
            "changes_made": affected_rows,
            "change_description": change_desc,
            "chunks_created": len(text_chunks),
            "text_length": len(extracted_text),
            "timestamp": timestamp,
            "version_number": new_version,
            "storage": "target_list + history_table + Pinecone (all linked by doc_id)"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing document: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing document: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.post("/search")
async def search_documents(query: str, top_k: int = 5, table_name: str = None):
    """
    Search for documents using semantic similarity
    
    Args:
        query: Search query text
        top_k: Number of results to return (default: 5)
        table_name: Optional filter by table name
    
    Returns:
        List of matching chunks with scores
    """
    try:
        # Generate embedding for search query
        query_embedding = embedding_model.encode([query]).tolist()[0]
        
        # Prepare filter if table_name is provided
        filter_dict = {"table_name": {"$eq": table_name}} if table_name else None
        
        # Search in Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        # Format results
        matches = []
        for match in results.get("matches", []):
            matches.append({
                "chunk_id": match["id"],
                "doc_id": match.get("metadata", {}).get("doc_id", ""),
                "score": match.get("score", 0),
                "metadata": match.get("metadata", {}),
                "text": match.get("metadata", {}).get("chunk_text", "")
            })
        
        return {
            "query": query,
            "matches_found": len(matches),
            "results": matches
        }
    
    except Exception as e:
        print(f"❌ Error searching documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching documents: {str(e)}"
        )


@router.get("/documents")
async def list_documents(table_name: str = None):
    """
    List all uploaded documents from history_table
    
    Args:
        table_name: Optional filter by table name
    
    Returns:
        List of uploaded documents with their changes
    """
    try:
        supabase = get_supabase_client()
        
        query = supabase.table("history_table").select("*")
        
        if table_name:
            query = query.eq("table_name", table_name)
        
        response = query.order("timestamp", desc=True).execute()
        
        return {
            "total_documents": len(response.data),
            "documents": response.data
        }
    
    except Exception as e:
        print(f"❌ Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing documents: {str(e)}"
        )


@router.get("/documents/{doc_id}")
async def get_document_details(doc_id: str):
    """
    Get complete details about a document
    Shows: history entry, Pinecone chunks, and what changes were made
    
    Args:
        doc_id: Document ID to retrieve
    
    Returns:
        Complete document information from both Supabase and Pinecone
    """
    supabase = get_supabase_client()
    
    try:
        # Get history entry from Supabase
        history = supabase.table("history_table").select("*") \
            .eq("doc_id", doc_id) \
            .execute()
        
        # Get chunks from Pinecone
        try:
            pinecone_results = index.query(
                vector=[0.0] * 384,  # Dummy vector for metadata retrieval
                top_k=100,
                filter={"doc_id": {"$eq": doc_id}},
                include_metadata=True
            )
            chunks = pinecone_results.get('matches', [])
        except:
            chunks = []
        
        return {
            "doc_id": doc_id,
            "history_entry": history.data[0] if history.data else None,
            "chunks_in_pinecone": len(chunks),
            "chunk_samples": [
                {
                    "chunk_id": c["id"],
                    "text_preview": c.get("metadata", {}).get("chunk_text", "")[:200]
                }
                for c in chunks[:3]  # Show first 3 chunks
            ],
            "status": "found" if history.data else "not found"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test-extraction")
async def test_extraction(file: UploadFile = File(...)):
    """
    TEST ENDPOINT: Extract and analyze document WITHOUT executing SQL
    
    This endpoint helps you verify:
    1. Text extraction is working
    2. LLM correctly identifies the action
    3. LLM extracts HCP data properly
    4. SQL query is generated correctly
    
    Use this to test before running the full upload flow
    
    Returns:
        - Extracted text
        - LLM extracted data
        - Generated SQL query
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        print(f"\n{'='*70}")
        print(f"🧪 TEST MODE - SQL Generation Preview")
        print(f"{'='*70}\n")
        
        # Step 1: Extract text
        print(f"1️⃣  Extracting text from {file.filename}...")
        extracted_text = extract_text_from_file(tmp_path, file.filename)
        print(f"   ✅ Extracted {len(extracted_text)} characters\n")
        
        # Step 2: Use LLM to extract data
        print(f"2️⃣  Analyzing with LLM...")
        extracted_data = extract_entities_with_llm(extracted_text)
        print(f"   Action: {extracted_data['action']}")
        print(f"   Subject: {extracted_data['hcp_data'].get('full_name', 'Unknown')}\n")
        
        # Step 3: Generate SQL
        print(f"3️⃣  Generating SQL...")
        sql = generate_sql_from_document(extracted_data)
        print(f"   ✅ SQL Generated\n")
        
        print(f"{'='*70}\n")
        
        return {
            "status": "test_mode",
            "message": "Analysis complete - SQL NOT executed",
            "filename": file.filename,
            "extracted_text_preview": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
            "extracted_text_length": len(extracted_text),
            "llm_analysis": {
                "action": extracted_data['action'],
                "hcp_data": extracted_data['hcp_data'],
                "reason": extracted_data.get('reason', ''),
                "identifier": extracted_data.get('identifier', '')
            },
            "generated_sql": sql,
            "note": "This SQL was NOT executed. Use /injection/upload to execute."
        }
    
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete a document and all its chunks from Pinecone and mark in history_table
    
    Args:
        doc_id: Document ID to delete
    
    Returns:
        Success message with deletion details
    """
    try:
        supabase = get_supabase_client()
        
        # Find all chunks in Pinecone for this document
        results = index.query(
            vector=[0] * 384,  # Dummy vector
            top_k=10000,
            include_metadata=True,
            filter={"doc_id": {"$eq": doc_id}}
        )
        
        # Delete chunks from Pinecone
        chunk_ids_to_delete = [match["id"] for match in results.get("matches", [])]
        if chunk_ids_to_delete:
            index.delete(ids=chunk_ids_to_delete)
            print(f"✅ Deleted {len(chunk_ids_to_delete)} chunks from Pinecone")
        
        # Mark as deleted in history_table (add new entry)
        version_resp = supabase.table("history_table").select("version_number") \
            .order("version_number", desc=True) \
            .limit(1) \
            .execute()
        
        last_version = version_resp.data[0]["version_number"] if version_resp.data else 0
        new_version = last_version + 1
        
        history_record = {
            "version_number": new_version,
            "operation_type": "DELETE",
            "table_name": "documents",
            "changed_rows": len(chunk_ids_to_delete),
            "reason": f"Document and chunks deleted: {doc_id}",
            "triggered_by": "system",
            "timestamp": datetime.utcnow().isoformat(),
            "doc_id": doc_id
        }
        
        supabase.table("history_table").insert(history_record).execute()
        
        return {
            "message": "Document deleted successfully",
            "doc_id": doc_id,
            "chunks_deleted": len(chunk_ids_to_delete),
            "history_updated": True
        }
    
    except Exception as e:
        print(f"❌ Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )