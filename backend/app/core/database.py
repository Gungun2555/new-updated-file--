
from .config import settings
try:
    from supabase import create_client
except Exception:
    create_client = None

SUPABASE_URL = settings.SUPABASE_URL
SUPABASE_KEY = settings.SUPABASE_KEY

def get_supabase_client():
    if create_client is None:
        raise RuntimeError("supabase-py not installed. Install with `pip install supabase`")
    return create_client(SUPABASE_URL, SUPABASE_KEY)
