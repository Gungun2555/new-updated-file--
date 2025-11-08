from langgraph.graph import StateGraph, START, END
from .nodes import (
    router_node,
    classify_query_node,
    generate_sql_node,
    execute_sql_node,
    fetch_versions_node,
    analyze_changes_node,
    summarizer_node
)
from .state import AgentState

def build_agent_graph():
    graph = StateGraph(AgentState)
    
    # Add all nodes
    graph.add_node("ROUTER", router_node)
    graph.add_node("CLASSIFY", classify_query_node)
    graph.add_node("GENERATE_SQL", generate_sql_node)
    graph.add_node("EXECUTE_SQL", execute_sql_node)
    graph.add_node("FETCH_VERSIONS", fetch_versions_node)
    graph.add_node("ANALYZE_CHANGES", analyze_changes_node)
    graph.add_node("SUMMARIZE", summarizer_node)
    
    # ✅ CRITICAL: Add edge from START to ROUTER 
    graph.add_edge(START, "ROUTER")
    
    # Router → Classify
    graph.add_edge("ROUTER", "CLASSIFY")
    
    # Conditional edge: if needs_sql is False, skip to SUMMARIZE
    def should_execute_sql(state):
        return "SQL_PATH" if state.get("needs_sql") else "CONVERSATION"
    
    graph.add_conditional_edges(
        "CLASSIFY",
        should_execute_sql,
        {
            "SQL_PATH": "GENERATE_SQL",
            "CONVERSATION": "SUMMARIZE"
        }
    )
    
    # SQL execution path
    graph.add_edge("GENERATE_SQL", "EXECUTE_SQL")
    graph.add_edge("EXECUTE_SQL", "SUMMARIZE")
    
    # Final edge to END
    graph.add_edge("SUMMARIZE", END)
    
    return graph.compile()
