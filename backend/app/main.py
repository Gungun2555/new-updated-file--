from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router as api_router
from app.core.config import settings
from app.core.database import get_supabase_client

app = FastAPI(title="Supabase FastAPI API with Chatbot Integration")

# ✅ Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173",
        "http://localhost:3000",  # React dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Root Route
@app.get("/")
def home():
    return {"message": "FastAPI + Supabase backend running successfully 🚀"}

# ✅ Include All Routers (CRUD + Lists + Injection + Chatbot)
app.include_router(api_router, prefix="/api")

# ✅ Optional: Check Supabase connection on startup
@app.on_event("startup")
def verify_supabase():
    try:
        client = get_supabase_client()
        print("✅ Supabase client initialized successfully.")
    except Exception as e:
        print(f"⚠️ Failed to initialize Supabase client: {e}")
