"""
Advanced Agent Nodes: LLM-Based Routing + Semantic Search with Pinecone
========================================================================
UPDATES:
- Replaced ChromaDB with Pinecone vector database
- Uses Pinecone for semantic search with embeddings
- Improved error handling to prevent bot crashes
- Better response formatting
"""

import os
import re
import json
import psycopg2
import psycopg2.extras
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
from typing import Optional, Dict, List, Any

from .state import AgentState
from .schema_context import SCHEMA_CONTEXT

# Pinecone for vector search
try:
    from pinecone import Pinecone
    from sentence_transformers import SentenceTransformer
    PINECONE_AVAILABLE = True
except Exception as e:
    print(f"⚠ Pinecone import error: {e}")
    PINECONE_AVAILABLE = False

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

# Pinecone Setup
pinecone_index = None
embedding_model = None

if PINECONE_AVAILABLE:
    try:
        from app.core.pinecone_client import pc, index
        pinecone_index = index
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Pinecone initialized for semantic search")
    except Exception as e:
        print(f"⚠ Pinecone initialization error: {e}")
        PINECONE_AVAILABLE = False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_sql(sql_text: str) -> str:
    """Remove markdown code fences and clean SQL text"""
    sql_text = re.sub(r"```(?:sql|SQL)?", "", sql_text)
    sql_text = re.sub(r"```", "", sql_text)
    return sql_text.strip()


def get_conversation_context(state: AgentState) -> str:
    """Generate context string from conversation history for LLM"""
    history = state.get("conversation_history", [])
    
    if not history:
        return "No previous context."
    
    lines = ["Recent conversation:"]
    for i, turn in enumerate(history[-3:], 1):
        query_text = turn.get('query', '')[:80]
        route_used = turn.get('type', 'unknown')
        lines.append(f"{i}. User asked: '{query_text}...' → Routed to: {route_used}")
    
    return "\n".join(lines)


def extract_changes_from_text(text: str) -> Dict[str, Any]:
    """Extract structured change information from document text using LLM"""
    prompt = f"""
    Analyze this document text and extract structured change information:
    
    TEXT:
    {text[:2000]}
    
    Extract and return JSON with this exact structure:
    {{
        "action": "added|removed|updated|modified|unknown",
        "items": ["specific item names extracted"],
        "count": <number of items changed>,
        "list_name": "name of the list/table/entity affected",
        "reason": "explanation for why this change was made",
        "timestamp": "date/time if mentioned, else null"
    }}
    
    Rules:
    - Return ONLY valid JSON, no markdown, no explanations
    - If information is not found, use null for that field
    - Be specific with item names, don't use placeholders
    - Extract the actual business reason for changes
    """
    
    try:
        response = model.generate_content(prompt)
        clean_text = response.text.strip()
        clean_text = clean_text.replace("json", "").replace("```", "").strip()
        result = json.loads(clean_text)
        return result
    except Exception as e:
        print(f"⚠ Could not extract changes: {e}")
        return {
            "action": None,
            "items": [],
            "count": 0,
            "list_name": None,
            "reason": None,
            "timestamp": None
        }


# =============================================================================
# LLM-BASED INTELLIGENT ROUTER
# =============================================================================

def router_node(state: AgentState) -> AgentState:
    """
    LLM-based intelligent routing - NO keyword matching!
    Uses context and intent understanding to route queries.
    """
    state_copy = dict(state)
    query = state_copy.get("user_query", "")
    history = state_copy.get("conversation_history", [])
    context_str = get_conversation_context(state_copy)
    
    routing_prompt = f"""
    You are an intelligent query router for a document management system. Analyze the user's query and conversation context to determine the best route.
    
    CONVERSATION CONTEXT:
    {context_str}
    
    CURRENT USER QUERY:
    "{query}"
    
    AVAILABLE ROUTES:
    
    1. database_only: Use when user wants to:
       - Query structured database tables (products, users, orders, etc.)
       - Get counts, lists, or specific records from tables
       - Filter or search database records
       - View metadata (uploader names, file names, timestamps)
       - Examples: "show all products", "how many users", "list orders by date"
    
    2. vector_store: Use when user wants to:
       - Search INSIDE document content (what documents say/contain)
       - Understand what changed in documents (additions, removals, updates)
       - Find why changes were made (reasons, explanations)
       - Semantic search across document text
       - Examples: "what was added to the call list", "why was this item removed", "what does the document say about pricing"
    
    3. hybrid: Use when user wants to:
       - Combine WHO uploaded with WHAT content (metadata + content)
       - Filter documents by uploader AND search their content
       - Find specific user's documents about a topic
       - Examples: "show documents uploaded by Aryan about pricing", "what did Gungun add to the target list"
    
    DECISION RULES:
    - If query references "those", "them", "these", "same" → inherit previous route from context
    - If query asks about WHO uploaded/modified → consider hybrid (unless only asking for metadata)
    - If query asks about WHAT is IN documents → use vector_store
    - If query asks for database records/counts → use database_only
    - When in doubt between vector_store and hybrid, choose vector_store (content search is more common)
    
    IMPORTANT:
    - Return ONLY valid JSON: {{"route": "database_only|vector_store|hybrid", "reasoning": "brief explanation"}}
    - No markdown, no code fences, no extra text
    - Be decisive - choose the single best route
    """
    
    try:
        response = model.generate_content(routing_prompt)
        clean_text = response.text.strip()
        clean_text = clean_text.replace("```json", "").replace("```", "").strip()
        
        routing_decision = json.loads(clean_text)
        route = routing_decision.get("route", "database_only")
        reasoning = routing_decision.get("reasoning", "No reasoning provided")
        
        # Validate route
        if route not in ["database_only", "vector_store", "hybrid"]:
            print(f"⚠ Invalid route '{route}', defaulting to database_only")
            route = "database_only"
        
        # Check Pinecone availability for routes that need it
        if route in ["vector_store", "hybrid"] and not PINECONE_AVAILABLE:
            print(f"⚠ {route} requested but Pinecone not available, falling back to database_only")
            route = "database_only"
        
        state_copy["route"] = route
        state_copy["routing_reasoning"] = reasoning
        
        print(f"\n{'='*60}")
        print(f"🎯 INTELLIGENT ROUTING DECISION")
        print(f"{'='*60}")
        print(f"Query: {query[:80]}...")
        print(f"Route: {route.upper()}")
        print(f"Reasoning: {reasoning}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"❌ Routing error: {e}, defaulting to database_only")
        state_copy["route"] = "database_only"
        state_copy["routing_reasoning"] = f"Error in routing: {str(e)}"
    
    return state_copy


# =============================================================================
# CLASSIFIER NODE
# =============================================================================

def classify_query_node(state: AgentState) -> AgentState:
    """Classify query type for better response formatting"""
    state_copy = dict(state)
    route = state_copy.get("route", "database_only")
    query = state_copy.get("user_query", "").lower()

    if route == "database_only":
        if "count" in query or "how many" in query:
            state_copy["query_type"] = "aggregate"
        elif "list" in query or "all" in query or "show" in query:
            state_copy["query_type"] = "listing"
        else:
            state_copy["query_type"] = "detail"
    elif route == "vector_store":
        state_copy["query_type"] = "semantic_search"
    elif route == "hybrid":
        state_copy["query_type"] = "hybrid_search"

    return state_copy


# =============================================================================
# DATABASE QUERY PATH
# =============================================================================

def generate_and_execute_sql(state: AgentState) -> AgentState:
    """Generate SQL using LLM with schema context"""
    state_copy = dict(state)
    query = state_copy.get("user_query", "")
    context_str = get_conversation_context(state_copy)

    sql_prompt = f"""
    You are a PostgreSQL expert. Generate a valid SQL query based on the schema and user request.
    
    DATABASE SCHEMA:
    {SCHEMA_CONTEXT}
    
    CONVERSATION CONTEXT:
    {context_str}
    
    USER REQUEST:
    {query}
    
    REQUIREMENTS:
    - Generate ONLY a valid PostgreSQL SELECT query
    - Use table/column names EXACTLY as shown in schema
    - Always add LIMIT 500 to prevent huge result sets
    - Use proper JOINs if multiple tables are needed
    - Handle NULL values appropriately
    - Return ONLY the SQL query, no markdown, no explanations
    
    SQL QUERY:
    """

    try:
        response = model.generate_content(sql_prompt)
        sql = clean_sql(response.text)
        
        state_copy["sql"] = sql
        print(f"📝 Generated SQL:\n{sql}\n")
    except Exception as e:
        print(f"❌ SQL generation error: {e}")
        state_copy["sql"] = None
        state_copy["error"] = f"SQL generation failed: {str(e)}"
        return state_copy

    # Execute SQL
    if not SUPABASE_DB_URL or not sql:
        state_copy["results"] = []
        state_copy["response"] = "Database not configured or SQL generation failed."
        state_copy["results_count"] = 0
        return state_copy

    conn = None
    try:
        conn = psycopg2.connect(SUPABASE_DB_URL)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            state_copy["results"] = [dict(r) for r in rows]
            state_copy["results_count"] = len(rows)
            print(f"✅ Retrieved {len(rows)} rows from database")
    except Exception as e:
        print(f"❌ SQL execution error: {e}")
        state_copy["results"] = []
        state_copy["error"] = f"Database query failed: {str(e)}"
        state_copy["results_count"] = 0
    finally:
        if conn:
            conn.close()

    return state_copy


# =============================================================================
# VECTOR STORE QUERY PATH - PINECONE SEMANTIC SEARCH
# =============================================================================

def query_vector_store(state: AgentState) -> AgentState:
    """
    Semantic search using Pinecone embeddings.
    Uses vector similarity to find relevant documents.
    """
    state_copy = dict(state)
    query = state_copy.get("user_query", "")
    
    print(f"\n{'='*60}")
    print(f"🔍 SEMANTIC SEARCH IN PINECONE")
    print(f"{'='*60}")
    print(f"Query: {query}")
    
    if not PINECONE_AVAILABLE or not embedding_model or not pinecone_index:
        print("❌ Pinecone not available")
        state_copy["results"] = []
        state_copy["response"] = "Vector search is not available."
        state_copy["results_count"] = 0
        return state_copy
    
    try:
        # Step 1: Generate query embedding
        print("📊 Generating query embedding...")
        query_embedding = embedding_model.encode([query]).tolist()[0]
        
        # Step 2: Perform semantic search in Pinecone
        print("🔎 Performing semantic similarity search in Pinecone...")
        search_results = pinecone_index.query(
            vector=query_embedding,
            top_k=20,
            include_metadata=True
        )
        
        if not search_results or not search_results.get('matches'):
            print("📭 No results found")
            state_copy["results"] = []
            state_copy["response"] = "No relevant documents found."
            state_copy["results_count"] = 0
            return state_copy
        
        # Step 3: Process results and group by document
        print(f"✅ Found {len(search_results['matches'])} relevant chunks")
        
        doc_chunks_map = {}
        for match in search_results['matches']:
            metadata = match.get('metadata', {})
            similarity = match.get('score', 0)
            chunk_text = match.get('values', {}).get('text', '')
            
            doc_id = metadata.get('doc_id')
            if not doc_id:
                continue
            
            if doc_id not in doc_chunks_map:
                doc_chunks_map[doc_id] = {
                    "doc_id": doc_id,
                    "filename": metadata.get('filename', 'Unknown'),
                    "uploader": metadata.get('uploader_name', 'Unknown'),
                    "table_name": metadata.get('table_name'),
                    "timestamp": metadata.get('timestamp'),
                    "chunks": [],
                    "max_similarity": similarity
                }
            
            doc_chunks_map[doc_id]["chunks"].append({
                "text": chunk_text or metadata.get('chunk_text', ''),
                "similarity": similarity,
                "metadata": metadata
            })
            
            if similarity > doc_chunks_map[doc_id]["max_similarity"]:
                doc_chunks_map[doc_id]["max_similarity"] = similarity
        
        # Step 4: Sort documents by relevance and extract changes
        print("📊 Extracting structured information from top documents...")
        sorted_docs = sorted(
            doc_chunks_map.values(),
            key=lambda x: x["max_similarity"],
            reverse=True
        )
        
        results = []
        for doc_data in sorted_docs[:5]:
            combined_text = "\n\n".join([c["text"] for c in doc_data["chunks"][:5]])
            changes = extract_changes_from_text(combined_text)
            
            result = {
                "doc_id": doc_data["doc_id"],
                "filename": doc_data["filename"],
                "uploader": doc_data["uploader"],
                "table_name": doc_data.get("table_name", ""),
                "timestamp": doc_data.get("timestamp", ""),
                "relevance_score": doc_data["max_similarity"],
                "action": changes.get("action"),
                "items_count": changes.get("count", 0),
                "items": changes.get("items", []),
                "list_name": changes.get("list_name"),
                "reason": changes.get("reason"),
                "top_chunks": doc_data["chunks"][:3]
            }
            results.append(result)
        
        state_copy["results"] = results
        state_copy["results_count"] = len(results)
        
        print(f"✅ Processed {len(results)} documents with semantic search")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"❌ Vector search error: {e}")
        import traceback
        traceback.print_exc()
        state_copy["results"] = []
        state_copy["response"] = "An error occurred during semantic search. Please try again."
        state_copy["results_count"] = 0
    
    return state_copy


# =============================================================================
# HYBRID QUERY PATH - PINECONE + METADATA
# =============================================================================

def hybrid_query(state: AgentState) -> AgentState:
    """
    Hybrid search using Pinecone semantic search + metadata filtering.
    LLM extracts filters → Database filters → Semantic similarity search
    """
    state_copy = dict(state)
    query = state_copy.get("user_query", "")
    
    print(f"\n{'='*60}")
    print(f"🔀 HYBRID SEARCH: Metadata + Semantic Content (Pinecone)")
    print(f"{'='*60}")
    print(f"Query: {query}")
    
    if not SUPABASE_DB_URL or not PINECONE_AVAILABLE or not embedding_model:
        print("❌ Hybrid search requirements not met")
        state_copy["results"] = []
        state_copy["response"] = "Hybrid search is not available."
        state_copy["results_count"] = 0
        return state_copy

    # Step 1: LLM extracts metadata filters
    filter_prompt = f"""
    Extract metadata filters from this user query for document search:
    
    QUERY: "{query}"
    
    Extract and return JSON:
    {{
        "uploader": "name of uploader/user if mentioned, else null",
        "table_name": "table/list name if mentioned, else null",
        "days_ago": "number of days if time period mentioned (0=today, 1=yesterday, 7=last week), else null",
        "search_term": "the main topic/content to search for (this will be used for semantic search)"
    }}
    
    Return ONLY valid JSON, no markdown, no explanations.
    """

    try:
        filter_response = model.generate_content(filter_prompt)
        filter_text = filter_response.text.strip()
        filter_text = filter_text.replace("```json", "").replace("```", "").strip()
        params = json.loads(filter_text)
        print(f"📋 Extracted filters: {json.dumps(params, indent=2)}")
    except Exception as e:
        print(f"⚠ Filter extraction failed: {e}, using query as search term")
        params = {"search_term": query}

    # Step 2: Query metadata table with filters
    sql = "SELECT doc_id, filename, uploader_name, table_name, timestamp FROM history_table WHERE operation_type = 'INSERT'"
    sql_params = []

    if params.get("uploader"):
        uploader = params["uploader"].strip().lower()
        sql += " AND LOWER(triggered_by) LIKE LOWER(%s)"
        sql_params.append(f"%{uploader}%")
        print(f"🔍 Filter: uploader contains '{uploader}'")

    if params.get("table_name"):
        table = params["table_name"].strip().lower()
        sql += " AND LOWER(table_name) LIKE LOWER(%s)"
        sql_params.append(f"%{table}%")
        print(f"🔍 Filter: table_name contains '{table}'")

    if params.get("days_ago") is not None:
        days = params["days_ago"]
        if days == 0:
            sql += " AND timestamp >= CURRENT_DATE"
            print(f"🔍 Filter: documents from today")
        else:
            sql += " AND timestamp >= NOW() - INTERVAL '%s days'"
            sql_params.append(days)
            print(f"🔍 Filter: documents from last {days} days")

    sql += " ORDER BY timestamp DESC LIMIT 100;"

    try:
        conn = psycopg2.connect(SUPABASE_DB_URL)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, sql_params)
            metadata_rows = [dict(r) for r in cur.fetchall()]
        conn.close()

        print(f"📚 Metadata filter returned {len(metadata_rows)} documents")

        if not metadata_rows:
            state_copy["results"] = []
            state_copy["response"] = "No documents match the metadata filters."
            state_copy["results_count"] = 0
            return state_copy

        # Step 3: Use Pinecone semantic search with filters
        doc_ids = [row['doc_id'] for row in metadata_rows]
        search_term = params.get("search_term", query)
        
        print(f"🔎 Performing SEMANTIC search for: '{search_term}'")
        print(f"   Searching within {len(doc_ids)} filtered documents")
        
        # Generate embedding for search term
        query_embedding = embedding_model.encode([search_term]).tolist()[0]

        # Perform semantic similarity search with doc_id filter
        search_results = pinecone_index.query(
            vector=query_embedding,
            top_k=min(50, len(doc_ids) * 3),
            filter={"doc_id": {"$in": doc_ids}},
            include_metadata=True
        )

        # Step 4: Combine and format results
        combined_results = []
        if search_results and search_results.get('matches'):
            for match in search_results['matches']:
                metadata = match.get('metadata', {})
                similarity = match.get('score', 0)
                chunk_text = match.get('values', {}).get('text', '')
                
                doc_id = metadata.get('doc_id')
                db_metadata = next((m for m in metadata_rows if m['doc_id'] == doc_id), {})

                combined_results.append({
                    "chunk_text": chunk_text or metadata.get('chunk_text', ''),
                    "metadata": metadata,
                    "filename": db_metadata.get('filename', 'Unknown'),
                    "uploader": db_metadata.get('uploader_name', 'Unknown'),
                    "table_name": db_metadata.get('table_name', ''),
                    "timestamp": db_metadata.get('timestamp', ''),
                    "relevance_score": similarity
                })
        
        # Sort by relevance
        combined_results.sort(key=lambda x: x['relevance_score'], reverse=True)

        state_copy["results"] = combined_results
        state_copy["results_count"] = len(combined_results)
        
        print(f"✅ Hybrid search complete: {len(combined_results)} relevant chunks")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"❌ Hybrid query error: {e}")
        import traceback
        traceback.print_exc()
        state_copy["results"] = []
        state_copy["response"] = "An error occurred during hybrid search. Please try again."
        state_copy["results_count"] = 0

    return state_copy


# =============================================================================
# RESPONSE FORMATTER
# =============================================================================

def format_response(state: AgentState) -> AgentState:
    """Format results into natural conversation response"""
    state_copy = dict(state)
    route = state_copy.get("route", "database_only")
    query_type = state_copy.get("query_type", "detail")
    results = state_copy.get("results", [])
    
    if state_copy.get("response"):
        return state_copy

    # =========== DATABASE ONLY RESPONSES ===========
    if route == "database_only":
        if not results:
            response = "📭 No results found in database."
        else:
            row_count = len(results)
            
            if query_type == "aggregate":
                if row_count == 1 and len(results[0]) == 1:
                    value = list(results[0].values())[0]
                    response = f"Result: {value}"
                else:
                    response = f"Found {row_count} records"
            else:
                response = f"📊 Database Results: {row_count} records\n\n"
                
                if row_count <= 10 and results:
                    headers = list(results[0].keys())[:5]
                    response += "| " + " | ".join(headers) + " |\n"
                    response += "|" + "|".join(["---"] * len(headers)) + "|\n"
                    
                    for row in results:
                        values = [str(row.get(h, ""))[:40] for h in headers]
                        response += "| " + " | ".join(values) + " |\n"
                else:
                    for i, row in enumerate(results[:20], 1):
                        response += f"{i}. {str(row)[:150]}...\n"
                    if row_count > 20:
                        response += f"\n... and {row_count - 20} more records"

    # =========== VECTOR STORE RESPONSES (Semantic Search) ===========
    elif route == "vector_store":
        if not results:
            response = "📭 No relevant documents found."
        else:
            response = f"📝 Semantic Search Results: {len(results)} documents\n\n"
            
            for i, doc in enumerate(results, 1):
                filename = doc.get("filename", "Unknown")
                uploader = doc.get("uploader", "Unknown")
                relevance = int(doc.get("relevance_score", 0) * 100)
                action = doc.get("action")
                items_count = doc.get("items_count", 0)
                list_name = doc.get("list_name")
                reason = doc.get("reason")
                
                response += f"{i}. {filename} ({relevance}% relevant)\n"
                response += f"   👤 Uploaded by: {uploader}\n"
                
                if action:
                    response += f"   ✏ Action: {action.upper()} {items_count} items"
                    if list_name:
                        response += f" to {list_name}"
                    response += "\n"
                
                if reason and reason != "No reason provided":
                    response += f"   💬 Reason: {reason}\n"
                
                chunks = doc.get("top_chunks", [])
                if chunks:
                    response += f"   📄 Relevant content:\n"
                    for chunk in chunks[:2]:
                        text = chunk.get("text", "")[:200]
                        chunk_sim = int(chunk.get("similarity", 0) * 100)
                        response += f"      - [{chunk_sim}%] {text}...\n"
                
                response += "\n"

    # =========== HYBRID RESPONSES ===========
    elif route == "hybrid":
        if not results:
            response = "📭 No documents match your search criteria."
        else:
            response = f"🔍 Hybrid Search: {len(results)} matches\n"
            response += "Filtered by metadata + semantic content search\n\n"
            
            docs_map = {}
            for result in results:
                filename = result.get('filename', 'Unknown')
                if filename not in docs_map:
                    docs_map[filename] = {
                        'uploader': result.get('uploader', 'Unknown'),
                        'chunks': [],
                        'max_relevance': result.get('relevance_score', 0)
                    }
                docs_map[filename]['chunks'].append({
                    'text': result.get('chunk_text', ''),
                    'relevance': result.get('relevance_score', 0)
                })
                docs_map[filename]['max_relevance'] = max(
                    docs_map[filename]['max_relevance'],
                    result.get('relevance_score', 0)
                )
            
            sorted_docs = sorted(
                docs_map.items(),
                key=lambda x: x[1]['max_relevance'],
                reverse=True
            )
            
            for i, (filename, data) in enumerate(sorted_docs[:10], 1):
                uploader = data['uploader']
                relevance = int(data['max_relevance'] * 100)
                
                response += f"{i}. {filename} (by {uploader}, {relevance}% relevant)\n"
                
                top_chunks = sorted(data['chunks'], key=lambda x: x['relevance'], reverse=True)[:2]
                for chunk in top_chunks:
                    text = chunk['text'][:250]
                    chunk_rel = int(chunk['relevance'] * 100)
                    response += f"   [{chunk_rel}%] {text}...\n"
                response += "\n"

    state_copy["response"] = response
    return state_copy


if __name__ == "__main__":
    print("✅ Advanced LLM-based agent nodes with Pinecone loaded successfully")