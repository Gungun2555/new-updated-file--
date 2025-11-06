from typing import TypedDict, List, Optional, Dict

class Message(TypedDict):
    """Represents a single message in conversation history"""
    role: str  # "user" or "assistant"
    content: str
    query_type: Optional[str]

class AgentState(TypedDict, total=False):
    user_query: str
    request_id: Optional[int]
    query_type: Optional[str]
    generated_sql: Optional[str]
    sql_params: Optional[Dict[str, any]]
    versions: Optional[List[Dict[str, any]]]
    comparisons: Optional[List[Dict[str, any]]]
    most_dynamic: Optional[Dict[str, any]]
    history_rows: Optional[List[Dict[str, any]]]
    attribution_rows: Optional[List[Dict[str, any]]]
    current_version: Optional[Dict[str, any]]
    rows: Optional[List[Dict[str, any]]]
    response: Optional[str]
    
    # 🆕 Enhanced conversation memory
    conversation_history: List[Message]
    context_summary: Optional[str]
    
    # 🆕 Session context (persistent across turns)
    session_context: Dict  # {
                          #   "current_table": str,
                          #   "last_query_type": str,
                          #   "active_request_id": int,
                          #   "last_results_summary": str,
                          #   "mentioned_tables": [str]
                          # }
    
    # 🆕 Memory management
    session_memory: Dict  # {
                         #   "conversation_context": str,
                         #   "query_history": list,
                         #   "result_cache": dict,
                         #   "entity_references": dict,
                         #   "last_topic": str,
                         #   "turn_count": int
                         # }
    
    # 🆕 Determine if SQL is needed
    needs_sql: bool  # If False, respond directly without SQL
    is_clarification: bool  # If True, user is asking follow-up about previous result
