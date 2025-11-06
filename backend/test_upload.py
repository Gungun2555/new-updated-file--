"""
Test script to find database constraints for all tables
"""
import sys
import os

# Add the backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.database import get_supabase_client

def test_table_constraints():
    """Test different values to find valid constraints"""
    sb = get_supabase_client()
    
    # Get the first list
    lists = sb.table('list_requests').select('*').limit(1).execute()
    if not lists.data or len(lists.data) == 0:
        print("No lists found in database")
        return
    
    list_id = lists.data[0]['request_id']
    
    # Create a test version
    version_data = {
        'request_id': list_id,
        'version_number': 999,
        'change_type': 'Test',
        'change_rationale': 'Testing constraints',
        'created_by': 'Test Script',
        'is_current': False
    }
    
    version_resp = sb.table('list_versions').insert(version_data).execute()
    version_id = version_resp.data[0]['version_id']
    
    print("\n" + "="*60)
    print("TESTING DATABASE CONSTRAINTS")
    print("="*60)
    
    # Test formulary_decision_maker_entries - influence_level
    print("\n1. Testing formulary_decision_maker_entries - influence_level field:")
    test_values = ['High', 'Medium', 'Low', 'Very High']
    for val in test_values:
        try:
            resp = sb.table('formulary_decision_maker_entries').insert({
                'version_id': version_id,
                'contact_id': 'TEST',
                'contact_name': 'Test',
                'organization': 'Test Org',
                'email': 'test@test.com',
                'influence_level': val
            }).execute()
            if resp.data:
                entry_id = resp.data[0]['entry_id']
                sb.table('formulary_decision_maker_entries').delete().eq('entry_id', entry_id).execute()
                print(f"   '{val}' - VALID")
        except Exception as e:
            if 'check constraint' in str(e).lower():
                print(f"   '{val}' - REJECTED by constraint")
    
    # Test competitor_target_entries - conversion_potential
    print("\n2. Testing competitor_target_entries - conversion_potential field:")
    test_values = ['High', 'Medium', 'Low', 'Very High']
    for val in test_values:
        try:
            resp = sb.table('competitor_target_entries').insert({
                'version_id': version_id,
                'hcp_id': 'TEST',
                'hcp_name': 'Test',
                'specialty': 'Test',
                'territory': 'Test',
                'competitor_product': 'Test',
                'conversion_potential': val,
                'assigned_rep': 'Test'
            }).execute()
            if resp.data:
                entry_id = resp.data[0]['entry_id']
                sb.table('competitor_target_entries').delete().eq('entry_id', entry_id).execute()
                print(f"   '{val}' - VALID")
        except Exception as e:
            if 'check constraint' in str(e).lower():
                print(f"   '{val}' - REJECTED by constraint")
    
    # Clean up
    sb.table('list_versions').delete().eq('version_id', version_id).execute()
    print("\n" + "="*60)

if __name__ == "__main__":
    test_table_constraints()
