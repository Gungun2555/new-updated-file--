"""
State Machine - 3 Path Execution Flow
=====================================
START → ROUTER → CLASSIFY → [3 Paths] → FORMAT → END
"""

from langgraph.graph import StateGraph, START, END
from .nodes import (
    router_node,
    classify_query_node,
    generate_and_execute_sql,
    query_vector_store,
    hybrid_query,
    format_response
)
from .state import AgentState


def build_agent_graph():
    """
    Build state machine for 3-path routing:
    1. database_only → SQL execution
    2. vector_store → Document change retrieval
    3. hybrid → Metadata filter + vector search
    """
    
    graph = StateGraph(AgentState)
    
    # =========================================================
    # ADD NODES
    # =========================================================
    graph.add_node("ROUTER", router_node)
    graph.add_node("CLASSIFY", classify_query_node)
    graph.add_node("EXECUTE_DB", generate_and_execute_sql)
    graph.add_node("QUERY_VECTOR", query_vector_store)
    graph.add_node("HYBRID_SEARCH", hybrid_query)
    graph.add_node("FORMAT", format_response)
    
    # =========================================================
    # EDGES
    # =========================================================
    
    # Entry flow: START → ROUTER → CLASSIFY
    graph.add_edge(START, "ROUTER")
    graph.add_edge("ROUTER", "CLASSIFY")
    
    # =========================================================
    # CONDITIONAL ROUTING after CLASSIFY
    # =========================================================
    def route_to_execution(state: AgentState) -> str:
        """Route to execution node based on route determined by ROUTER"""
        route = state.get("route", "database_only")
        
        if route == "database_only":
            return "EXECUTE_DB"
        elif route == "vector_store":
            return "QUERY_VECTOR"
        elif route == "hybrid":
            return "HYBRID_SEARCH"
        else:
            return "EXECUTE_DB"
    
    graph.add_conditional_edges(
        "CLASSIFY",
        route_to_execution,
        {
            "EXECUTE_DB": "EXECUTE_DB",
            "QUERY_VECTOR": "QUERY_VECTOR",
            "HYBRID_SEARCH": "HYBRID_SEARCH"
        }
    )
    
    # =========================================================
    # ALL EXECUTION PATHS CONVERGE TO FORMAT
    # =========================================================
    graph.add_edge("EXECUTE_DB", "FORMAT")
    graph.add_edge("QUERY_VECTOR", "FORMAT")
    graph.add_edge("HYBRID_SEARCH", "FORMAT")
    
    # =========================================================
    # FORMAT → END
    # =========================================================
    graph.add_edge("FORMAT", END)
    
    # =========================================================
    # COMPILE & RETURN
    # =========================================================
    return graph.compile()


if __name__ == "__main__":
    graph = build_agent_graph()
    print("✅ Agent graph built successfully")