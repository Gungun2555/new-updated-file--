"""
Complete Agent Nodes with Advanced Memory & Context Management
================================================================
Features:
- Smart clarification detection
- Session memory with context tracking
- Conversation summarization for long chats
- Reference to previous results
- Multi-turn query understanding
- Gemini API Integration (replacing OpenAI)
"""

import os
import re
import json
import psycopg2
import psycopg2.extras
import google.generativeai as genai
from .state import AgentState
from .schema_context import SCHEMA_CONTEXT
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ✅ Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")


# ═══════════════════════════════════════════════════════════════
# MEMORY MANAGEMENT UTILITIES
# ═══════════════════════════════════════════════════════════════

def initialize_memory(state: AgentState) -> AgentState:
    """Initialize or retrieve session memory"""
    if not state.get("session_memory"):
        state["session_memory"] = {
            "conversation_context": "",
            "query_history": [],
            "result_cache": {},
            "entity_references": {},
            "last_topic": None,
            "last_table": None,
            "turn_count": 0
        }
    return state


def update_memory(state: AgentState) -> AgentState:
    """Update session memory after each turn"""
    state = initialize_memory(state)
    memory = state.get("session_memory", {})
    
    if "query_history" not in memory:
        memory["query_history"] = []
    if "result_cache" not in memory:
        memory["result_cache"] = {}
    if "entity_references" not in memory:
        memory["entity_references"] = {}
    
    memory["turn_count"] = memory.get("turn_count", 0) + 1
    
    query = state.get("user_query", "")
    sql = state.get("generated_sql")
    rows = state.get("rows", [])
    response = state.get("response", "")
    
    query_record = {
        "turn": memory["turn_count"],
        "query": query,
        "sql": sql,
        "row_count": len(rows),
        "timestamp": datetime.now().isoformat(),
        "response_summary": response[:200] if response else ""
    }
    
    memory["query_history"].append(query_record)
    
    if len(memory["query_history"]) > 10:
        memory["query_history"] = memory["query_history"][-10:]
    
    if sql and rows:
        cache_key = f"turn_{memory['turn_count']}"
        memory["result_cache"][cache_key] = {
            "sql": sql,
            "rows": rows[:50],
            "full_count": len(rows)
        }
        
        if 'FROM' in sql.upper():
            table_match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
            if table_match:
                memory["last_table"] = table_match.group(1)
                state["session_context"]["current_table"] = table_match.group(1)
    
    entities = extract_entities(query, rows)
    memory["entity_references"].update(entities)
    
    if state.get("query_type"):
        memory["last_topic"] = state["query_type"]
    
    if memory["turn_count"] % 3 == 0:
        memory["conversation_context"] = summarize_conversation_context(memory["query_history"])
    
    state["session_memory"] = memory
    return state


def extract_entities(query: str, rows: list) -> dict:
    """Extract mentioned entities from query and results"""
    entities = {}
    
    tables = ["list_versions", "target_list_entries", "hcp", "version"]
    for table in tables:
        if table in query.lower():
            entities[f"table_{table}"] = True
    
    if rows and isinstance(rows[0], dict):
        for row in rows[:5]:
            for key, value in row.items():
                if 'name' in key.lower() and isinstance(value, str):
                    entities[f"name_{value}"] = True
    
    return entities


def summarize_conversation_context(query_history: list) -> str:
    """Generate a summary of recent conversation for context"""
    if not query_history:
        return "No previous conversation."
    
    recent = query_history[-5:]
    summary_parts = []
    
    for record in recent:
        summary_parts.append(
            f"Turn {record['turn']}: Asked about '{record['query'][:60]}...' "
            f"→ {record['row_count']} results"
        )
    
    return "\n".join(summary_parts)


# ═══════════════════════════════════════════════════════════════
# ROUTER NODE WITH MEMORY AWARENESS
# ═══════════════════════════════════════════════════════════════

def router_node(state: AgentState) -> AgentState:
    """
    Smart router using memory to understand context and intent.
    Distinguishes between new queries, clarifications, and follow-ups.
    """
    state = initialize_memory(state)
    
    query = state.get("user_query", "").lower()
    memory = state.get("session_memory", {})
    query_history = memory.get("query_history", [])
    
    if not state.get("session_context"):
        state["session_context"] = {
            "current_table": None,
            "last_query_type": None,
            "active_request_id": state.get("request_id"),
            "last_results_summary": "",
            "mentioned_tables": [],
            "last_sql_query": None,
            "last_result_count": 0
        }
    
    new_query_indicators = [
        "give me", "show me", "retrieve", "fetch", "get me", "find",
        "i want", "i need", "can you get", "can you show", "list",
        "another question", "new question", "different question",
        "from table", "from the", "select", "query", "what are", "what is"
    ]
    
    clarification_indicators = [
        "about them", "about these", "about those", "about it", "about that",
        "the same", "those ones", "these ones", "from that", "from those",
        "tell me more", "more about", "more details", "more info",
        "what about", "how about", "why", "explain"
    ]
    
    reference_indicators = [
        "the results", "the data", "those results", "that list",
        "the previous", "last query", "before", "earlier"
    ]
    
    is_new_query = any(indicator in query for indicator in new_query_indicators)
    is_clarification = any(indicator in query for indicator in clarification_indicators)
    has_reference = any(indicator in query for indicator in reference_indicators)
    
    sql_keywords = [
        "show", "list", "find", "search", "get", "count", "how many",
        "compare", "difference", "version", "when", "created", "modified",
        "sql", "query", "table", "data", "record", "entry", "hcp", "all",
        "retrieve", "fetch", "select", "from", "where", "entries", "what are",
        "give", "display", "who", "which", "lists", "targets", "requests",
        "revenue", "tier", "specialty", "address", "phone", "email", "city",
        "state", "zip", "npi", "prescriber", "value", "total", "sum"
    ]
    needs_sql_keywords = any(kw in query for kw in sql_keywords)
    
    has_entity_reference = False
    if memory.get("entity_references"):
        for entity in memory["entity_references"].keys():
            if entity.lower() in query:
                has_entity_reference = True
                needs_sql_keywords = True
                break
    
    pure_greetings = ["hello", "hi", "hey", "thanks", "thank you", "bye", "good morning", "good evening"]
    is_pure_greeting = query.strip() in pure_greetings
    
    meta_questions = ["how do you work", "what can you do", "help me", "how does this work"]
    is_meta = any(m in query for m in meta_questions)
    
    needs_sql = False
    is_true_clarification = False
    
    if is_pure_greeting or is_meta:
        needs_sql = False
        print("🎯 GREETING/META: No SQL execution")
    else:
        needs_sql = True
        is_true_clarification = False
        print(f"🎯 FORCING SQL EXECUTION: query='{query[:50]}'")
    
    tables = ["list_versions", "target_list_entries", "version", "entry", "hcp", "list"]
    for table in tables:
        if table in query:
            if table not in state["session_context"]["mentioned_tables"]:
                state["session_context"]["mentioned_tables"].append(table)
            state["session_context"]["current_table"] = table
    
    state["needs_sql"] = needs_sql
    state["is_clarification"] = is_true_clarification
    
    print(f"🔀 Router Decision: needs_sql={needs_sql}, is_clarification={is_true_clarification}")
    return state


# ═══════════════════════════════════════════════════════════════
# CLASSIFY QUERY NODE (Gemini)
# ═══════════════════════════════════════════════════════════════

def classify_query_node(state: AgentState) -> AgentState:
    """Classify query type with memory context using Gemini"""
    
    if not state.get("needs_sql"):
        state["query_type"] = "conversation"
        return state
    
    state = initialize_memory(state)
    
    query = state.get("user_query", "")
    session_ctx = state.get("session_context", {})
    memory = state.get("session_memory", {})
    
    recent_context = ""
    if memory.get("query_history"):
        recent = memory["query_history"][-3:]
        recent_context = "Recent queries: " + ", ".join([q["query"][:40] for q in recent])
    
    prompt = f"""
    Classify this query into ONE of:
    version_comparison, history, attribution, current_state, list_all, ad_hoc_select

    Context table: {session_ctx.get('current_table', 'unknown')}
    {recent_context}
    Query: {query}
    
    Return only one word category.
    """

    response = model.generate_content(prompt)
    category = response.text.strip().lower()
    state["query_type"] = category
    state["session_context"]["last_query_type"] = category
    print(f"🧭 Query classified as → {category}")
    return state


# ═══════════════════════════════════════════════════════════════
# GENERATE SQL NODE (Gemini) WITH MEMORY
# ═══════════════════════════════════════════════════════════════

def generate_sql_node(state: AgentState) -> AgentState:
    """Generate SQL with full awareness of conversation history using Gemini"""
    
    if not state.get("needs_sql"):
        state["generated_sql"] = None
        return state
    
    state = initialize_memory(state)
    
    user_query = state.get("user_query", "")
    session_ctx = state.get("session_context", {})
    memory = state.get("session_memory", {})
    
    context_info = f"""
    Current table/topic: {session_ctx.get('current_table', 'unknown')}
    Last table queried: {memory.get('last_table', 'unknown')}
    Previously mentioned tables: {', '.join(session_ctx.get('mentioned_tables', []))}
    """
    
    last_sql = None
    last_query_text = None
    last_table = None
    
    if memory.get("query_history"):
        context_info += "\n\nRecent queries in this conversation:\n"
        for record in memory["query_history"][-5:]:
            context_info += f"- Turn {record['turn']}: {record['query'][:80]}\n"
            if record['sql']:
                context_info += f"  SQL: {record['sql'][:100]}\n"
                last_sql = record['sql']
                last_query_text = record['query']
                if 'FROM' in record['sql'].upper():
                    table_match = re.search(r'FROM\s+(\w+)', record['sql'], re.IGNORECASE)
                    if table_match:
                        last_table = table_match.group(1)
        
        if last_sql and last_query_text:
            context_info += f"""
    
    ⚠️ MOST RECENT QUERY (use this as primary context):
    User asked: "{last_query_text}"
    SQL executed: {last_sql}
    Table used: {last_table or 'unknown'}
    
    If the current question refers to "them", "those entries", "full entry", "more details",
    it likely means the results from the above query.
    """
    
    if memory.get("entity_references"):
        entities = list(memory["entity_references"].keys())[:10]
        context_info += f"\nMentioned entities: {', '.join(entities)}\n"

    prompt = f"""
    You are an expert SQL generator for PostgreSQL (Supabase) database.

    Database schema:
    {SCHEMA_CONTEXT}

    Conversation context:
    {context_info}
    
    Conversation summary:
    {memory.get('conversation_context', 'First query in session')}

    Current user question: {user_query}

    ⚠️ CRITICAL CONTEXT AWARENESS:
    The MOST RECENT table queried was: {last_table or memory.get('last_table', 'N/A')}
    The MOST RECENT SQL executed was: {last_sql or 'N/A'}

    CRITICAL INSTRUCTIONS:
    1. Generate a **single SELECT SQL query** (no DML/DDL).
    
    2. **CONTEXT MATCHING RULES** (HIGHEST PRIORITY):
       - If user mentions a PERSON NAME (like "Dr. Nikhil Kapoor") that appeared in recent results:
         → Query the SAME table from the most recent query ({last_table})
         → Use WHERE clause to filter for that specific person
       - If user says "give details for X", "show only X", "filter for X":
         → Query the SAME table from: {last_table}
         → Add WHERE condition for X
       - If user asks "give full entry", "show details", "more info":
         → Query the SAME table: {last_table}
         → Use SELECT * to show ALL columns
    
    3. **TABLE SELECTION**:
       - If question references "them", "those", "the previous ones": Use {last_table}
       - If question mentions a specific list type: Use the appropriate table
       - If uncertain: Default to {last_table}
    
    4. **FIELD NAME MATCHING**:
       - For idn_health_system_entries: contact_name field contains person's name
       - For target_list_entries: hcp_name field contains person's name
       - Use ILIKE '%name%' for flexible name matching
    
    5. Return valid PostgreSQL SQL only. No explanations, no markdown, no code blocks.
    """

    response = model.generate_content(prompt)
    sql_query = response.text.strip()
    
    # Clean markdown artifacts from Gemini
    sql_query = re.sub(r"^```(?:sql)?", "", sql_query, flags=re.IGNORECASE | re.MULTILINE)
    sql_query = re.sub(r"```$", "", sql_query, flags=re.MULTILINE)
    sql_query = sql_query.strip()

    state["generated_sql"] = sql_query
    state["session_context"]["last_sql_query"] = sql_query
    print("🧩 Generated SQL Query:\n", sql_query)
    return state


# ═══════════════════════════════════════════════════════════════
# EXECUTE SQL NODE
# ═══════════════════════════════════════════════════════════════

def execute_sql_node(state: AgentState) -> AgentState:
    """Execute SQL and store results in memory"""
    
    if not state.get("needs_sql") or not state.get("generated_sql"):
        state["rows"] = []
        return state

    sql = state.get("generated_sql")
    conn = None
    
    try:
        print(f"🔗 Connecting to Supabase...")
        conn = psycopg2.connect(SUPABASE_DB_URL)
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            state["rows"] = [dict(r) for r in rows]
            print(f"📊 Retrieved {len(rows)} rows")
            
            state["session_context"]["last_results_summary"] = f"Retrieved {len(rows)} rows"
            state["session_context"]["last_result_count"] = len(rows)
            
    except Exception as e:
        state["rows"] = []
        print("❌ SQL Execution Error:", str(e))
        state["session_context"]["last_results_summary"] = f"Error: {str(e)}"
        
    finally:
        if conn is not None:
            conn.close()

    return state


# ═══════════════════════════════════════════════════════════════
# FETCH VERSIONS NODE
# ═══════════════════════════════════════════════════════════════

def fetch_versions_node(state: AgentState) -> AgentState:
    """Fetch version history for analysis"""
    rid = state.get("request_id")
    if rid is None:
        print("⚠️ No request_id provided — skipping version fetch.")
        state["versions"] = []
        return state

    sql = """
    SELECT id AS version_id, version_number, created_at
    FROM list_versions
    WHERE request_id = %s
    ORDER BY version_number ASC;
    """

    conn = None
    try:
        conn = psycopg2.connect(SUPABASE_DB_URL)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (rid,))
            rows = cur.fetchall()
            state["versions"] = [dict(r) for r in rows]
            print(f"📄 Found {len(rows)} versions for request_id={rid}")
    except Exception as e:
        print("❌ Error fetching versions:", e)
        state["versions"] = []
    finally:
        if conn:
            conn.close()

    return state


# ═══════════════════════════════════════════════════════════════
# ANALYZE CHANGES NODE
# ═══════════════════════════════════════════════════════════════

def analyze_changes_node(state: AgentState) -> AgentState:
    """Analyze version changes"""
    versions = state.get("versions", [])
    if not versions:
        print("⚠️ No versions found — skipping analysis.")
        state["most_dynamic"] = {"version_number": None, "total": 0}
        return state

    comparisons = []
    most_dynamic = {"version_number": None, "total": -1}

    conn = None
    try:
        conn = psycopg2.connect(SUPABASE_DB_URL)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            for i in range(1, len(versions)):
                prev_id = versions[i - 1]["version_id"]
                curr_id = versions[i]["version_id"]

                cur.execute("SELECT hcp_id FROM target_list_entries WHERE version_id = %s", (prev_id,))
                prev_hcps = {r["hcp_id"] for r in cur.fetchall()}

                cur.execute("SELECT hcp_id FROM target_list_entries WHERE version_id = %s", (curr_id,))
                curr_hcps = {r["hcp_id"] for r in cur.fetchall()}

                added = curr_hcps - prev_hcps
                removed = prev_hcps - curr_hcps
                total = len(added) + len(removed)

                comparisons.append({
                    "from_version": versions[i - 1]["version_number"],
                    "to_version": versions[i]["version_number"],
                    "added": len(added),
                    "removed": len(removed),
                    "total": total
                })

                if total > most_dynamic["total"]:
                    most_dynamic = {
                        "version_number": versions[i]["version_number"],
                        "total": total
                    }
    except Exception as e:
        print("❌ Error analyzing changes:", e)
    finally:
        if conn:
            conn.close()

    state["comparisons"] = comparisons
    state["most_dynamic"] = most_dynamic
    print("🔍 Most dynamic version:", most_dynamic)
    return state


# ═══════════════════════════════════════════════════════════════
# SMART SUMMARIZER WITH MEMORY (Gemini)
# ═══════════════════════════════════════════════════════════════

def summarizer_node(state: AgentState) -> AgentState:
    """
    Generate intelligent responses using full conversation memory with Gemini.
    Shows all results when appropriate, provides context-aware summaries.
    """
    
    state = initialize_memory(state)
    
    query = state.get("user_query", "")
    query_type = state.get("query_type", "conversation")
    rows = state.get("rows", [])
    is_clarification = state.get("is_clarification", False)
    memory = state.get("session_memory", {})
    session_ctx = state.get("session_context", {})

    conversation_history = state.get("conversation_history", [])
    
    history_text = memory.get("conversation_context", "")
    if not history_text and conversation_history:
        history_text = "Recent conversation:\n"
        for msg in conversation_history[-4:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content'][:100]}...\n"

    # STRATEGY 1: Direct display for list queries
    if query_type in ["list_all", "ad_hoc_select"] and rows:
        row_count = len(rows)
        
        if row_count <= 100:
            if rows and isinstance(rows[0], dict):
                keys = list(rows[0].keys())
                
                display_fields = []
                priority_fields = ['hcp_name', 'name', 'system_name', 'title', 'specialty', 
                                 'contact_name', 'system_id', 'hcp_id', 'tier', 'importance']
                
                for field in priority_fields:
                    if field in keys:
                        display_fields.append(field)
                
                if not display_fields:
                    display_fields = [k for k in keys if k not in ['id', 'created_at', 'updated_at', 'version_id']][:5]
                
                field_labels = {
                    'hcp_name': 'Name', 'name': 'Name', 'system_name': 'System',
                    'title': 'Title', 'specialty': 'Specialty', 'contact_name': 'Contact',
                    'system_id': 'ID', 'hcp_id': 'HCP ID', 'tier': 'Tier',
                    'importance': 'Importance', 'contact_email': 'Email',
                    'phone': 'Phone', 'address': 'Address', 'city': 'City',
                    'state': 'State', 'npi': 'NPI', 'revenue': 'Revenue',
                    'prescriber_type': 'Type'
                }
                
                results_list = []
                for i, row in enumerate(rows):
                    row_parts = []
                    for field in display_fields:
                        value = row.get(field)
                        if value:
                            label = field_labels.get(field, field.replace('_', ' ').title())
                            row_parts.append(f"{label}: {value}")
                    
                    if row_parts:
                        results_list.append(f"{i+1}. {' | '.join(row_parts)}")
                
                results_text = "\n".join(results_list)
                response = f"Here are all {row_count} entries:\n\n{results_text}"
                state["response"] = response
                print(f"\n🗣️ Displaying all {row_count} results with details.\n")
                
                state = update_memory(state)
                return state
        
        else:
            sample_size = 20
            if rows and isinstance(rows[0], dict):
                keys = list(rows[0].keys())
                
                display_fields = []
                priority_fields = ['hcp_name', 'name', 'system_name', 'title', 'specialty', 
                                 'contact_name', 'system_id', 'tier']
                for field in priority_fields:
                    if field in keys:
                        display_fields.append(field)
                
                if not display_fields:
                    display_fields = [k for k in keys if k not in ['id', 'created_at', 'updated_at']][:4]
                
                field_labels = {
                    'hcp_name': 'Name', 'name': 'Name', 'system_name': 'System',
                    'title': 'Title', 'specialty': 'Specialty', 'contact_name': 'Contact',
                    'system_id': 'ID', 'tier': 'Tier', 'importance': 'Importance',
                    'contact_email': 'Email', 'revenue': 'Revenue'
                }
                
                sample_list = []
                for i, row in enumerate(rows[:sample_size]):
                    row_parts = []
                    for field in display_fields:
                        value = row.get(field)
                        if value:
                            label = field_labels.get(field, field.replace('_', ' ').title())
                            row_parts.append(f"{label}: {value}")
                    if row_parts:
                        sample_list.append(f"{i+1}. {' | '.join(row_parts)}")
                
                sample_text = "\n".join(sample_list)
                
                response = f"""Found {row_count} entries in total.

Here are the first {sample_size}:

{sample_text}

... and {row_count - sample_size} more entries.

Would you like me to show a specific range or filter these results?"""
                
                state["response"] = response
                print(f"\n🗣️ Large dataset: showing {sample_size}/{row_count} results.\n")
                
                state = update_memory(state)
                return state

    # STRATEGY 2: Use Gemini for summarization
    else:
        if rows:
            data_summary = json.dumps(rows[:5], indent=2, default=str)
        else:
            data_summary = "No data."

        prompt = f"""
        You are a helpful assistant summarizing database query results.

        Query: {query}
        Query type: {query_type}
        Data retrieved: {data_summary}
        
        Conversation context:
        {history_text}

        Provide a clear and natural response to the user's query based on the data.
        """

        response = model.generate_content(prompt)
        summary = response.text.strip()
        state["response"] = summary
        print("\n🗣️ Response generated with Gemini & memory context.\n")

        state = update_memory(state)
        return state