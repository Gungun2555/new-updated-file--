from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Dict, Any, Optional
import google.generativeai as genai
from supabase import create_client
import os
from dotenv import load_dotenv
from openai import OpenAI

# 🧩 Load environment variables
load_dotenv()

# --- INITIALIZATION ---
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configure ChatAI/OpenAI client
client = OpenAI(api_key=os.getenv("CHATAI_API_KEY"))

# --- STATE ---
class RAGState(TypedDict):
    question: str
    chat_history: List[Dict[str, str]]
    query_embedding: List[float]
    retrieved_docs: List[Dict[str, Any]]
    context_text: str
    final_answer: str
    last_retrieved_content: str  # Store last retrieved docs for follow-ups


# Generate Embedding with Context
def embed_query(state: RAGState):
    """Generate embedding considering conversation context."""
    try:
        question_text = state["question"]
        
        # If it's a follow-up question, combine with last question for better context
        if state.get("chat_history") and len(state["chat_history"]) > 0:
            last_exchange = state["chat_history"][-1]
            # Check if current question is short (likely a follow-up)
            if len(question_text.split()) < 5:
                combined_query = f"{last_exchange['user']} {question_text}"
                question_text = combined_query
        
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=question_text,
            task_type="retrieval_query"
        )
        state["query_embedding"] = response["embedding"]
    except Exception as e:
        print(f"Error in embed_query: {e}")
        state["query_embedding"] = []
    return state


# Retrieve from Supabase
def retrieve_docs(state: RAGState):
    """Retrieve documents with smart filtering."""
    try:
        query_vec = state["query_embedding"]
        if not query_vec:
            state["retrieved_docs"] = []
            return state

        result = supabase.rpc(
            "match_list_embeddings_simple",
            {
                "query_embedding": query_vec,
                "match_threshold": 0.35,
                "match_count": 8
            },
        ).execute()

        all_docs = result.data or []
        
        # Filter for relevance
        relevant_docs = [d for d in all_docs if d.get('similarity', 0) > 0.4]
        state["retrieved_docs"] = relevant_docs

    except Exception as e:
        print(f"Error in retrieve_docs: {e}")
        state["retrieved_docs"] = []
    return state


# Compose Context Intelligently
def compose_context(state: RAGState):
    """Smart context composition with memory."""
    try:
        context_parts = []
        
        # Add recent conversation history (last 2 exchanges)
        if state.get("chat_history") and len(state["chat_history"]) > 0:
            recent_history = state["chat_history"][-2:]
            history_text = "\n".join([
                f"Previous Q: {msg['user']}\nPrevious A: {msg['assistant']}"
                for msg in recent_history
            ])
            context_parts.append(f"=== CONVERSATION HISTORY ===\n{history_text}\n")
        
        # Add newly retrieved documents OR use cached content for follow-ups
        if state["retrieved_docs"]:
            docs_text = "\n\n---\n\n".join([
                f"[Document {i+1} - Similarity: {r.get('similarity', 0):.2f}]\n{r.get('content', '')}"
                for i, r in enumerate(state["retrieved_docs"][:3])
            ])
            context_parts.append(f"=== RELEVANT INFORMATION ===\n{docs_text}\n")
            # Cache this for follow-ups
            state["last_retrieved_content"] = docs_text
        elif state.get("last_retrieved_content"):
            # Use cached content for follow-up questions
            context_parts.append(f"=== RELEVANT INFORMATION (from previous query) ===\n{state['last_retrieved_content']}\n")
        
        if context_parts:
            state["context_text"] = "\n".join(context_parts) + f"\n=== CURRENT QUESTION ===\n{state['question']}"
        else:
            state["context_text"] = f"No relevant information available.\n\nQuestion: {state['question']}"
            
    except Exception as e:
        print(f"Error in compose_context: {e}")
        state["context_text"] = state["question"]
    return state


# Generate Answer with Better Instructions
def generate_answer(state: RAGState):
    """Generate intelligent, context-aware answers."""
    try:
        system_prompt = """You are a helpful pharmaceutical data assistant with excellent memory.

CRITICAL RULES:
1. ALWAYS use conversation history to understand pronouns and references:
   - "he/she/they" → refer to people mentioned before
   - "that/it/this" → refer to things just discussed
   - "why/reason" → explain the last topic discussed
   - "when/where" → provide details about last mentioned item

2. When asked about specific people (e.g., "Dr. Rohan", "Rahul"):
   - Search the documents for that exact name
   - If found, provide ALL available details (email, phone, specialty, etc.)
   - If not found, clearly say "I don't have information about [name] in the current data"

3. Answer style:
   - Be conversational and natural
   - Don't repeat information already shared unless asked
   - If the question is a follow-up, directly answer without restating previous info
   - Keep answers concise (2-4 sentences)

4. If no relevant info exists, say: "I don't have that information in the current context."
"""

        user_prompt = f"""{state['context_text']}

Answer the current question naturally, using conversation history to understand context."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=250
        )
        
        state["final_answer"] = response.choices[0].message.content
        
    except Exception as e:
        print(f"Error in generate_answer: {e}")
        state["final_answer"] = "I encountered an error. Please try again."
    return state


# Build the LangGraph RAG Pipeline
def build_rag_graph():
    builder = StateGraph(RAGState)
    builder.add_node("embed_query", embed_query)
    builder.add_node("retrieve_docs", retrieve_docs)
    builder.add_node("compose_context", compose_context)
    builder.add_node("generate_answer", generate_answer)

    builder.add_edge(START, "embed_query")
    builder.add_edge("embed_query", "retrieve_docs")
    builder.add_edge("retrieve_docs", "compose_context")
    builder.add_edge("compose_context", "generate_answer")
    builder.add_edge("generate_answer", END)
    return builder.compile()


# Run Interactive Loop
if __name__ == "__main__":
    graph = build_rag_graph()
    chat_history = []
    last_retrieved_content = ""

    print("=" * 70)
    print("Pharmaceutical Data Assistant")
    print("=" * 70)
    print("Ask me anything! Type 'exit' to quit, 'clear' to reset memory\n")

    while True:
        user_question = input("You: ").strip()
        
        if user_question.lower() in ["exit", "quit", "bye"]:
            print("\nGoodbye!")
            break
        
        if user_question.lower() == "clear":
            chat_history = []
            last_retrieved_content = ""
            print("Memory cleared!\n")
            continue
        
        if not user_question:
            continue

        initial_state = {
            "question": user_question,
            "chat_history": chat_history,
            "query_embedding": [],
            "retrieved_docs": [],
            "context_text": "",
            "final_answer": "",
            "last_retrieved_content": last_retrieved_content
        }

        final_state = graph.invoke(initial_state)

        answer = final_state.get('final_answer', 'No answer generated')
        print(f"Assistant: {answer}\n")
        
        # Update chat history
        chat_history.append({
            "user": user_question,
            "assistant": answer
        })
        
        # Keep last 3 exchanges
        if len(chat_history) > 3:
            chat_history = chat_history[-3:]
        
        # Update cached content
        last_retrieved_content = final_state.get("last_retrieved_content", last_retrieved_content)