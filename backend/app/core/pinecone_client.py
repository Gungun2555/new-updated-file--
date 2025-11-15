"""
Pinecone client initialization and configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from pinecone import Pinecone
    
    # Get credentials from environment
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
    index_name = os.getenv("PINECONE_INDEX_NAME", "document-store")
    
    if not api_key:
        raise ValueError("❌ PINECONE_API_KEY not found in environment variables")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=api_key, environment=environment)
    
    # Get index
    index = pc.Index(index_name)
    
    print(f"✅ Pinecone client initialized")
    print(f"   Index: {index_name}")
    print(f"   Environment: {environment}")
    
except ImportError as e:
    print(f"❌ Pinecone not installed: {e}")
    print("   Run: pip install pinecone")
    raise
except Exception as e:
    print(f"❌ Pinecone initialization error: {e}")
    raise