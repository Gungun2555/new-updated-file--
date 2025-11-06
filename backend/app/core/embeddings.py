import os
import google.generativeai as genai
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Setup Supabase ---
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# --- Setup Gemini ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Helper Function to Safely Fetch Data ---
def safe_fetch(table_name, filter_col=None, filter_val=None):
    query = supabase.table(table_name).select("*")
    if filter_col and filter_val:
        query = query.eq(filter_col, filter_val)
    response = query.execute()
    return response.data if response.data else []

# --- Generate Combined Text for a Version ---
def build_version_content(version):
    version_id = version["version_id"]
    # Ensure safe_fetch returns a list or is handled for indexing (e.g., [0])
    request = safe_fetch("list_requests", "request_id", version["request_id"])
    
    # Safely access elements with a check for 'request' first
    subdomain = safe_fetch("subdomains", "subdomain_id", request[0]["subdomain_id"]) if request and request[0].get("subdomain_id") else []
    
    # Safely access elements with a check for 'subdomain' first
    domain = safe_fetch("domains", "domain_id", subdomain[0]["domain_id"]) if subdomain and subdomain[0].get("domain_id") else []

    # Collect all entry tables
    entry_tables = [
        "call_list_entries",
        "target_list_entries",
        "event_invitation_entries",
        "formulary_decision_maker_entries",
        "digital_engagement_entries",
        "idn_health_system_entries",
        "high_value_prescriber_entries",
        "work_logs"  # included logs as context
    ]

    entries_summary = []

    for table in entry_tables:
        rows = safe_fetch(table, "version_id", version_id)
        if not rows:
            continue

        # Convert each row into readable text
        formatted = [", ".join([f"{k}: {v}" for k, v in row.items() if v is not None]) for row in rows]
        entries_summary.append(f"Table: {table}\n" + "\n".join(formatted))

    entries_summary_content = (
        '\n\n'.join(entries_summary) 
        if entries_summary 
        else 'No entries for this version.'
    )

    # Combine all parts
    content = f"""
Domain: {domain[0]['domain_name'] if domain else 'N/A'}
Subdomain: {subdomain[0]['subdomain_name'] if subdomain else 'N/A'}
Request Purpose: {request[0]['request_purpose'] if request else 'N/A'}
Version: {version['version_number']} | Change: {version['change_type']}
Created By: {version['created_by']}
Change Rationale: {version['change_rationale']}

Entries Summary:
{entries_summary_content}
"""
    return content.strip()


# Main Embedding Generator 
def generate_embeddings_for_all_versions():
    versions = safe_fetch("list_versions")

    print(f"Found {len(versions)} list versions to process...\n")

    for v in versions:
        try:
            print(f"🧩 Processing version ID {v['version_id']} ...")
            content = build_version_content(v)

            # Generate embedding
            embedding_response = genai.embed_content(
                model="models/text-embedding-004",
                content=content,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embedding = embedding_response["embedding"]

            # Upsert into embeddings table
            supabase.table("list_embeddings").upsert({
                "entity_type": "list_version",
                "entity_id": v["version_id"],
                "version_id": v["version_id"],
                "content": content,
                "embedding": embedding
            }).execute()

            print(f"Stored embedding for version {v['version_id']}\n")

        except Exception as e:
            print(f"Error processing version {v.get('version_id', 'N/A')}: {e}")

if __name__ == "__main__":
    generate_embeddings_for_all_versions()
