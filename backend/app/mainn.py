"""
Multi-turn Conversational Database Query Agent
Main entry point for the agent application
"""

from state_machine import build_agent_graph, create_initial_state
from dotenv import load_dotenv
import os

load_dotenv()


def format_conversation_context(conversation_history: list) -> str:
    """
    Formats conversation history into readable context for memory.
    Keeps last 6 messages (3 conversation turns) for context.
    """
    if not conversation_history:
        return "No previous conversation."
    
    context = "Recent conversation:\n"
    for msg in conversation_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg.get('content', '')[:150]
        context += f"\n{role}: {content}...\n"
    
    return context


def validate_environment():
    """Validate that required environment variables are set"""
    required_vars = ["GEMINI_API_KEY"]
    optional_vars = ["SUPABASE_DB_URL"]
    
    missing_required = []
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    if missing_required:
        print("❌ Missing required environment variables:")
        for var in missing_required:
            print(f"   - {var}")
        return False
    
    # Warn if optional vars are missing
    missing_optional = []
    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)
    
    if missing_optional:
        print("⚠️  Warning: Optional environment variables not set:")
        for var in missing_optional:
            print(f"   - {var} (database operations will fail)")
    
    return True


def main():
    """Main conversation loop"""
    
    print("=" * 70)
    print("🤖 Conversational Database Query Agent")
    print("=" * 70)
    print("💡 Tip: Ask natural questions or request specific SQL queries")
    print("📝 Type 'exit', 'quit', 'bye', 'done', 'q' to end\n")
    
    # Validate environment
    if not validate_environment():
        print("\n❌ Setup incomplete. Please configure .env file.")
        return
    
    # Build the agent graph
    try:
        graph = build_agent_graph()
        print("✅ Agent initialized successfully\n")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # Session-level state that persists across turns
    conversation_history = []
    session_memory = {
        "conversation_context": "",
        "query_history": [],
        "result_cache": {},
        "entity_references": {},
        "last_topic": None,
        "turn_count": 0
    }
    
    # Get optional request_id from user
    print("Session Configuration")
    print("-" * 70)
    request_id = None
    rid_input = input("Optional request_id for this session (press Enter to skip): ").strip()
    
    if rid_input:
        try:
            request_id = int(rid_input)
            print(f"✅ Using request_id: {request_id}\n")
        except ValueError:
            print("⚠️  Invalid request_id format, skipping...\n")
            request_id = None
    
    # Conversation loop
    turn = 0
    print("=" * 70)
    print("Starting conversation. Type your query below:")
    print("=" * 70)
    
    while True:
        turn += 1
        print(f"\n[Turn {turn}]")
        
        try:
            user_query = input("📝 You: ").strip()
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except EOFError:
            print("\n\n👋 EOF reached. Goodbye!")
            break
        
        # Exit conditions
        if user_query.lower() in ['exit', 'quit', 'bye', 'done', 'q']:
            print("\n👋 Thanks for chatting! Goodbye.")
            break
        
        # Empty query handling
        if not user_query:
            print("⚠️  Please enter a query.")
            continue
        
        # Create initial state with all required fields
        try:
            initial_state = create_initial_state(
                user_query=user_query,
                request_id=request_id
            )
            
            # Update with conversation history and memory
            initial_state["conversation_history"] = conversation_history
            initial_state["context_summary"] = format_conversation_context(conversation_history)
            initial_state["session_memory"] = session_memory
            
        except Exception as e:
            print(f"❌ Error preparing state: {e}")
            continue
        
        # Invoke the graph
        try:
            print("⏳ Processing query...")
            final_state = graph.invoke(initial_state)
            
            # Extract response
            response = final_state.get("response")
            if not response:
                response = "Sorry, I couldn't generate a response."
            
            # Print response
            print(f"\n🤖 Assistant: {response}")
            
            # Print metadata if available
            query_type = final_state.get("query_type")
            row_count = len(final_state.get("rows", []))
            
            if query_type:
                print(f"   [Query Type: {query_type}]", end="")
            if row_count > 0:
                print(f" [Results: {row_count} rows]", end="")
            if query_type or row_count > 0:
                print()
            
            # Update session memory for next turn
            session_memory = final_state.get("session_memory", session_memory)
            
            # Add to conversation history
            conversation_history.append({
                "role": "user",
                "content": user_query,
                "query_type": query_type
            })
            
            conversation_history.append({
                "role": "assistant",
                "content": response,
                "query_type": query_type
            })
            
            # Keep conversation history manageable (last 20 messages = 10 turns)
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]
        
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            print("\n📋 Debug: Attempting to show full error trace...")
            import traceback
            traceback.print_exc()
            print("\n💡 Tip: Check that all required environment variables are set.")
            continue


def interactive_test():
    """Quick test mode for debugging"""
    print("\n🧪 Running in TEST mode...\n")
    
    from state_machine import build_agent_graph, create_initial_state
    
    graph = build_agent_graph()
    
    # Test query
    initial_state = create_initial_state(
        user_query="List all HCPs",
        request_id=1
    )
    
    print("📊 Testing with: 'List all HCPs'\n")
    
    try:
        result = graph.invoke(initial_state)
        print("✅ Test successful!\n")
        print("Response:", result.get("response"))
        print("Query Type:", result.get("query_type"))
        print("SQL:", result.get("generated_sql"))
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    # Check for test mode flag
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        interactive_test()
    else:
        main()