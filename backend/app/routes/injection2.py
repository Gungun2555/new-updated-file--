"""
Smart Document Injection: Upload → Extract Metadata → Extract Text → Generate SQL → Execute → Store
Auto-extracts uploader info from document, no form fields needed
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


def extract_table_name_from_document(document_text: str) -> str:
    """
    Extract table name from the beginning of document
    Looks for "table:" or "Table:" on the first few lines
    """
    lines = document_text.split('\n')[:10]  # Check first 10 lines
    
    for line in lines:
        # Look for "table:" or "Table:" pattern
        match = re.search(r'table\s*:\s*(\w+)', line, re.IGNORECASE)
        if match:
            table_name = match.group(1).strip()
            print(f"   📍 Found table name in document: {table_name}")
            return table_name
    
    # Default to target_list if not found
    print(f"   📍 Table name not found in document, using default: target_list")
    return "target_list"


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
    ⚠️ DO NOT manually insert history records here - triggers handle it!
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
                match = re.search(r'\((.*?)\)\s+VALUES', sql, re.IGNORECASE)
                if match:
                    fields_str = match.group(1)
                    fields = [f.strip() for f in fields_str.split(',')]
                    
                    for field in fields:
                        params.append(hcp_data.get(field))
                    
                    print(f"   📍 Parsed {len(fields)} fields from SQL: {fields}")
                else:
                    for key in ['hcp_code', 'full_name', 'gender', 'qualification', 'specialty', 
                               'designation', 'email', 'phone', 'hospital_name', 'hospital_address',
                               'city', 'state', 'pincode', 'experience_years', 'influence_score',
                               'category', 'therapy_area', 'monthly_sales', 'yearly_sales',
                               'last_interaction_date', 'call_frequency', 'priority']:
                        params.append(hcp_data.get(key))
            
            elif action == 'UPDATE':
                set_match = re.search(r'SET\s+(.*?)\s+WHERE', sql, re.IGNORECASE)
                if set_match:
                    set_clause = set_match.group(1)
                    set_fields = [f.split('=')[0].strip() for f in set_clause.split(',')]
                    
                    for field in set_fields:
                        params.append(hcp_data.get(field))
                    
                    params.append(identifier)
                    print(f"   📍 Parsed {len(set_fields)} update fields: {set_fields}")
                else:
                    for key, value in hcp_data.items():
                        if value is not None:
                            params.append(value)
                    params.append(identifier)
            
            elif action == 'DELETE':
                params = [identifier]
            
            # Count placeholders in SQL
            placeholder_count = sql.count('%s')
            
            print(f"   📍 SQL has {placeholder_count} placeholders, providing {len(params)} parameters")
            
            if placeholder_count != len(params):
                print(f"   ⚠️  Parameter count mismatch! Adjusting...")
                if len(params) > placeholder_count:
                    params = params[:placeholder_count]
                elif len(params) < placeholder_count:
                    params.extend([None] * (placeholder_count - len(params)))
            
            # Execute query - trigger will handle history_table insert
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
    Smart Document Upload: Extract Table Name → Extract Text → Generate SQL → Execute → Store
    
    IMPORTANT: 
    - uploader_name provided as form field
    - table_name extracted from document (look for "Table: table_name" on first line)
    - All HCP data extracted from document body
    - Defaults to "target_list" if no table name found
    """
    
    supabase = get_supabase_client()
    timestamp = datetime.utcnow().isoformat()
    
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
        
        # Step 2: Extract table name from document
        print(f"2️⃣  Extracting table name from document...")
        table_name = extract_table_name_from_document(extracted_text)
        
        # Step 3: Use LLM to extract HCP structured data
        print(f"3️⃣  Analyzing document for HCP data...")
        extracted_data = extract_entities_with_llm(extracted_text)
        print(f"   Action: {extracted_data['action']}")
        print(f"   Subject: {extracted_data['hcp_data'].get('full_name', 'Unknown')}")
        
        # Step 4: Generate SQL query
        print(f"4️⃣  Generating SQL query...")
        sql = generate_sql_from_document(extracted_data)
        
        # Step 5: Execute SQL on target_list
        # ⚠️ Trigger will automatically insert into history_table
        print(f"5️⃣  Executing query on target_list...")
        success, affected_rows, change_desc = execute_sql_and_get_changes(sql, extracted_data)
        print(f"   ✅ {change_desc}")
        
        # Get the version_number that was auto-created by trigger
        print(f"6️⃣  Fetching auto-created history entry...")
        version_resp = supabase.table("history_table").select("version_number") \
            .eq("table_name", table_name) \
            .order("version_number", desc=True) \
            .limit(1) \
            .execute()
        
        version_number = version_resp.data[0]["version_number"] if version_resp.data else 1
        
        # Step 6: Generate doc_id
        print(f"7️⃣  Preparing document for storage...")
        doc_id = generate_document_id(uploader_name, file.filename, timestamp)
        
        # Split text into chunks
        text_chunks = chunk_text(extracted_text)
        
        # Generate embeddings for semantic search
        chunk_embeddings = embedding_model.encode(text_chunks).tolist()
        
        # Step 7: Store in Pinecone with metadata
        print(f"8️⃣  Storing in Pinecone...")
        vectors_to_upsert = []
        
        for i, (chunk, embedding) in enumerate(zip(text_chunks, chunk_embeddings)):
            chunk_id = f"{doc_id}_chunk_{i}"
            
            metadata_vector = {
                "doc_id": doc_id,
                "version_number": str(version_number),
                "uploader_name": uploader_name,
                "table_name": table_name,
                "action": extracted_data['action'],
                "hcp_name": extracted_data['hcp_data'].get('full_name', ''),
                "filename": file.filename,
                "file_type": os.path.splitext(file.filename)[1],
                "timestamp": timestamp,
                "chunk_index": str(i),
                "total_chunks": str(len(text_chunks)),
                "chunk_size": str(len(chunk)),
                "chunk_text": chunk[:500],
                "change_description": change_desc
            }
            
            vectors_to_upsert.append((chunk_id, embedding, metadata_vector))
        
        # Upload to Pinecone
        index.upsert(vectors=vectors_to_upsert)
        print(f"   ✅ Stored {len(vectors_to_upsert)} chunks in Pinecone")
        
        # Step 8: Update the AUTO-CREATED history entry with doc_id
        print(f"9️⃣  Linking doc_id to history entry...")
        
        supabase.table("history_table").update({
            "doc_id": doc_id,
            "filename": file.filename,
            "file_type": os.path.splitext(file.filename)[1],
            "num_chunks": len(text_chunks),
            "triggered_by": uploader_name,
            "reason": change_desc
        }).eq("version_number", version_number).eq("table_name", table_name).execute()
        
        print(f"   ✅ History entry updated with doc_id: {doc_id}")
        
        print(f"\n{'='*70}")
        print(f"✅ DOCUMENT PROCESSING COMPLETE")
        print(f"{'='*70}\n")
        
        return {
            "message": "Document uploaded and processed successfully",
            "doc_id": doc_id,
            "uploader_name": uploader_name,
            "table_name": table_name,
            "filename": file.filename,
            "action": extracted_data['action'],
            "subject": extracted_data['hcp_data'].get('full_name', 'Unknown'),
            "changes_made": affected_rows,
            "change_description": change_desc,
            "chunks_created": len(text_chunks),
            "text_length": len(extracted_text),
            "timestamp": timestamp,
            "version_number": version_number
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
    """Search for documents using semantic similarity"""
    try:
        query_embedding = embedding_model.encode([query]).tolist()[0]
        filter_dict = {"table_name": {"$eq": table_name}} if table_name else None
        
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
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
    """List all uploaded documents from history_table"""
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
    """Get complete details about a document"""
    supabase = get_supabase_client()
    
    try:
        history = supabase.table("history_table").select("*") \
            .eq("doc_id", doc_id) \
            .execute()
        
        try:
            pinecone_results = index.query(
                vector=[0.0] * 384,
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
                for c in chunks[:3]
            ],
            "status": "found" if history.data else "not found"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and all its chunks from Pinecone and mark in history_table"""
    try:
        supabase = get_supabase_client()
        
        results = index.query(
            vector=[0] * 384,
            top_k=10000,
            include_metadata=True,
            filter={"doc_id": {"$eq": doc_id}}
        )
        
        chunk_ids_to_delete = [match["id"] for match in results.get("matches", [])]
        if chunk_ids_to_delete:
            index.delete(ids=chunk_ids_to_delete)
            print(f"✅ Deleted {len(chunk_ids_to_delete)} chunks from Pinecone")
        
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