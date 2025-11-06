# SQL-Based Chatbot Integration - README

## 📍 Integration Overview

The SQL-based conversational chatbot has been successfully integrated into your pharma application. It provides natural language querying capabilities over your pharmaceutical database.

## 📂 Folder Structure

```
backend/
  app/
    chatbot/                    # NEW - Chatbot module
      __init__.py
      state.py                 # State management for conversation
      state_machine.py         # LangGraph workflow
      nodes.py                 # Agent nodes (router, SQL generator, executor, summarizer)
      schema_context.py        # Database schema documentation for AI
    routes/
      chatbot.py              # NEW - Chatbot API endpoints
      router.py               # UPDATED - Includes chatbot routes
    main.py                    # Existing (no changes needed)
  requirements.txt            # UPDATED - Added openai, psycopg2-binary

pharma-frontend/
  src/
    api/
      listbotApi.ts           # UPDATED - Points to new chatbot endpoint
    hooks/
      useListBotChat.ts       # UPDATED - Manages conversation state
    components/
      SidebarListBot.tsx      # UPDATED - New SQL-focused UI
```

## 🔧 Configuration Required

### 1. Environment Variables

Add to your `.env` file in the `backend/` directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
SUPABASE_DB_URL=postgresql://user:password@host:port/database
```

### 2. Install Dependencies

```powershell
cd backend
pip install -r requirements.txt
```

## 🚀 API Endpoints

### POST `/api/chatbot/query`
Main chatbot endpoint for conversational queries.

**Request:**
```json
{
  "question": "Show all HCPs in the database",
  "chat_history": [],
  "session_id": "default",
  "request_id": null
}
```

**Response:**
```json
{
  "answer": "Here are all 150 HCPs...",
  "generated_sql": "SELECT * FROM target_list_entries...",
  "row_count": 150,
  "query_type": "list_all"
}
```

### POST `/api/chatbot/clear-session`
Clears conversation history for a session.

### GET `/api/chatbot/health`
Health check endpoint.

## 🎯 Features

### 1. **Smart Routing**
- Detects whether a query needs SQL execution or is conversational
- Distinguishes between new queries and clarifications
- Maintains conversation context

### 2. **Memory Management**
- Session-based conversation history
- Entity tracking (tables, names, IDs mentioned)
- Result caching for follow-up questions

### 3. **SQL Generation**
- Generates PostgreSQL queries from natural language
- Uses comprehensive schema context
- Handles complex joins and filters

### 4. **Intelligent Summarization**
- Direct display for list queries (≤100 items)
- Sampling for large datasets
- Context-aware responses with conversation memory

## 💬 Example Queries

```
"Show all HCPs in the database"
"List all versions for request_id 1"
"Who made the most recent changes?"
"Count all target list entries"
"Show high value prescribers"
"What changed between version 1 and 2?"
"Tell me about the HCPs in cardiology"
"Who requested list version 3?"
```

## 🔄 How It Works

1. **User sends query** → Frontend (`SidebarListBot.tsx`)
2. **API call** → `POST /api/chatbot/query`
3. **Router node** → Determines if SQL is needed
4. **Classification** → Categorizes query type
5. **SQL Generation** → Creates SQL query with OpenAI
6. **Execution** → Runs query on Supabase
7. **Summarization** → Generates natural language response
8. **Response** → Returns to frontend with conversation context

## 🧪 Testing

1. Start backend server:
```powershell
cd backend
uvicorn app.main:app --reload
```

2. Start frontend:
```powershell
cd pharma-frontend
npm run dev
```

3. Open the application and use the SQL Assistant sidebar (right side)

## 🎨 UI Updates

The chatbot sidebar now features:
- **Title**: "SQL Assistant" instead of "AI Assistant"
- **No domain requirement**: Direct querying without domain selection
- **Sample questions**: Quick-start prompts
- **Better formatting**: Supports multi-line responses with `whitespace-pre-wrap`

## 🔐 Security Considerations

- SQL queries are read-only (SELECT only)
- Input validation prevents SQL injection
- Session management for conversation isolation
- API key protection via environment variables

## 📊 Database Schema Support

The chatbot understands:
- `domains`, `subdomains`
- `list_requests`, `list_versions`
- `target_list_entries`, `call_list_entries`
- `competitor_target_entries`
- `digital_engagement_entries`
- `formulary_decision_maker_entries`
- `high_value_prescriber_entries`
- `idn_health_system_entries`
- `work_logs`

Plus all views:
- `view_request_context`
- `view_target_list_full`
- `view_list_evolution`
- `v_current_state_target_list`
- `view_work_attribution`

## 🐛 Troubleshooting

### Chatbot not responding
- Check `OPENAI_API_KEY` is set correctly
- Verify backend server is running
- Check browser console for errors

### SQL errors
- Ensure `SUPABASE_DB_URL` is correct
- Verify database connection
- Check backend terminal for SQL query output

### Frontend errors
- Clear browser cache
- Check network tab for failed API calls
- Verify API endpoint URL in `axiosClient.ts`

## 🔮 Future Enhancements

- [ ] Redis-based session storage for production
- [ ] Export conversation history
- [ ] SQL query visualization
- [ ] User feedback on responses
- [ ] Custom domain-specific training
- [ ] Query optimization suggestions

## 📝 Notes

- Conversation history is stored in memory (resets on server restart)
- Maximum 20 messages kept per session
- SQL queries are limited to SELECT statements
- Large result sets are automatically sampled (first 20 items shown)

---

**Integration completed successfully! 🎉**
