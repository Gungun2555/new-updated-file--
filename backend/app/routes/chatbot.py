"""
Chatbot API Routes
Handles SQL-based conversational queries about the pharmaceutical database
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from app.chatbot.state_machine import build_agent_graph
from datetime import datetime

router = APIRouter(prefix="/chatbot", tags=["chatbot"])

# Build the agent graph once at module load
agent_graph = build_agent_graph()

# Session storage for conversation history (in production, use Redis or database)
conversation_sessions: Dict[str, List[Dict]] = {}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatbotRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]] = []
    request_id: Optional[int] = None
    session_id: Optional[str] = None


class ChatbotResponse(BaseModel):
    answer: str
    generated_sql: Optional[str] = None
    row_count: Optional[int] = 0
    query_type: Optional[str] = None


def format_conversation_context(conversation_history: list) -> str:
    """Formats conversation history into readable context"""
    if not conversation_history:
        return "No previous conversation."
    
    context = "Recent conversation:\n"
    for msg in conversation_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg['content'][:150]
        context += f"\n{role}: {content}...\n"
    
    return context


@router.post("/query", response_model=ChatbotResponse)
async def chat_query(request: ChatbotRequest):
    """
    Process a conversational database query using the SQL agent.
    
    The agent can:
    - Answer questions about lists, HCPs, versions, and changes
    - Maintain conversation context across turns
    - Generate and execute SQL queries automatically
    - Provide intelligent summaries and explanations
    """
    try:
        # Get or create session
        session_id = request.session_id or "default"
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = []
        
        conversation_history = conversation_sessions[session_id]
        
        # Build initial state
        initial_state = {
            "user_query": request.question,
            "conversation_history": conversation_history,
            "context_summary": format_conversation_context(conversation_history),
            "session_context": {
                "current_table": None,
                "last_query_type": None,
                "active_request_id": request.request_id,
                "last_results_summary": "",
                "mentioned_tables": [],
                "last_sql_query": None,
                "last_result_count": 0
            },
            "request_id": request.request_id,
            "needs_sql": False,
            "is_clarification": False
        }
        
        # Invoke the agent graph
        final_state = agent_graph.invoke(initial_state)
        
        response_text = final_state.get("response", "Sorry, I couldn't generate a response.")
        
        # Update conversation history
        conversation_history.append({
            "role": "user",
            "content": request.question,
            "query_type": final_state.get("query_type")
        })
        
        conversation_history.append({
            "role": "assistant",
            "content": response_text,
            "query_type": final_state.get("query_type")
        })
        
        # Keep conversation history manageable (last 20 messages)
        if len(conversation_history) > 20:
            conversation_sessions[session_id] = conversation_history[-20:]
        else:
            conversation_sessions[session_id] = conversation_history
        
        # Return response
        return ChatbotResponse(
            answer=response_text,
            generated_sql=final_state.get("generated_sql"),
            row_count=len(final_state.get("rows", [])),
            query_type=final_state.get("query_type")
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post("/clear-session")
async def clear_session(session_id: str = "default"):
    """Clear conversation history for a session"""
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]
    return {"message": f"Session {session_id} cleared"}


@router.get("/health")
async def health_check():
    """Check if the chatbot service is running"""
    return {
        "status": "healthy",
        "active_sessions": len(conversation_sessions),
        "timestamp": datetime.now().isoformat()
    }
