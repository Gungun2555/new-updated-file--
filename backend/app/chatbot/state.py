"""
Minimal LangGraph State - Only What's Used
===========================================
Stripped to essentials for 3-path routing:
database_only | vector_store | hybrid
"""

from typing import TypedDict, Optional, List, Dict, Any, Annotated
import operator


class AgentState(TypedDict, total=False):
    """
    Minimal state - only fields actually used in nodes
    """
    
    # Input (from user)
    user_query: str
    
    # Routing decision (set by router_node)
    route: str  # "database_only" | "vector_store" | "hybrid"
    
    # Query classification (set by classify_query_node)
    query_type: str  # "aggregate", "listing", "detail", "semantic_search", "hybrid_search"
    
    # Execution results (populated by execute nodes)
    results: List[Dict[str, Any]]
    
    # SQL query (if database path)
    sql: Optional[str]
    
    # Final response (set by format_response node)
    response: str
    
    # Memory - conversation history (last 5 turns, append-only)
    conversation_history: Annotated[List[Dict[str, Any]], operator.add]
    
    # Result count (for quick access)
    results_count: int
    
    # Error handling
    error: Optional[str]