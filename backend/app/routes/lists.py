"""
Custom Lists Router - Provides high-level list management endpoints
Maps to list_requests table in Supabase
"""
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from app.core.database import get_supabase_client
from typing import List, Dict, Any, Optional
from fastapi import UploadFile, File
import csv
from io import StringIO

router = APIRouter(prefix='/lists', tags=['lists'])

def _get_supabase():
    return get_supabase_client()

@router.get('', response_model=List[Dict[str, Any]])
def get_lists(category: Optional[str] = None, subdomain_id: Optional[int] = None, limit: int = 100):
    """
    Get all lists, optionally filtered by category or subdomain_id
    Maps to list_requests table with subdomain information
    Only returns lists that have actual entries (non-empty lists)
    """
    sb = _get_supabase()
    try:
        # Use a more efficient query to get lists with their subdomain info
        query = sb.table('list_requests').select('*, subdomains(*)')
        
        if subdomain_id is not None:
            query = query.eq('subdomain_id', subdomain_id)
        
        resp = query.limit(limit).order('created_at', desc=True).execute()
        lists = resp.data if hasattr(resp, 'data') else resp
        
        # Table mapping for subdomain entries
        table_mapping = {
            'Target Lists': 'target_list_entries',
            'Call Lists': 'call_list_entries',
            'Formulary Decision-Maker Lists': 'formulary_decision_maker_entries',
            'IDN/Health System Lists': 'idn_health_system_entries',
            'Event Invitation Lists': 'event_invitation_entries',
            'Digital Engagement Lists': 'digital_engagement_entries',
            'High-Value Prescriber Lists': 'high_value_prescriber_entries',
            'Competitor Target Lists': 'competitor_target_entries'
        }
        
        # Filter to only include lists with entries
        filtered_lists = []
        for list_item in lists:
            version_resp = sb.table('list_versions').select('*').eq('request_id', list_item['request_id']).eq('is_current', True).order('version_number', desc=True).limit(1).execute()
            if version_resp.data and len(version_resp.data) > 0:
                list_item['current_version'] = version_resp.data[0]
                
                # Check if this list has actual entries
                if list_item.get('subdomains'):
                    subdomain_name = list_item['subdomains']['subdomain_name']
                    entry_table = table_mapping.get(subdomain_name)
                    
                    if entry_table:
                        # Check if there are entries for this version
                        entries_resp = sb.table(entry_table).select('entry_id', count='exact').eq('version_id', version_resp.data[0]['version_id']).limit(1).execute()
                        # Only include this list if it has entries
                        if entries_resp.count and entries_resp.count > 0:
                            filtered_lists.append(list_item)
                    else:
                        # If no entry table mapping, include it anyway
                        filtered_lists.append(list_item)
                else:
                    # If no subdomain info, include it anyway
                    filtered_lists.append(list_item)
        
        return filtered_lists
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/{list_id}', response_model=Dict[str, Any])
def get_list_detail(list_id: int):
    """
    Get detailed information for a specific list
    Returns list_request with related data and current snapshot items
    """
    sb = _get_supabase()
    try:
        print(f"\n[DEBUG GET_LIST_DETAIL] Fetching list {list_id}")
        
        # Get the list request
        resp = sb.table('list_requests').select('*').eq('request_id', list_id).execute()
        data = resp.data if hasattr(resp, 'data') else resp
        
        if not data or len(data) == 0:
            raise HTTPException(status_code=404, detail='List not found')
        
        list_data = data[0]
        print(f"[DEBUG GET_LIST_DETAIL] Found list: {list_data.get('request_purpose')}")
        
        # Get subdomain info
        subdomain_resp = sb.table('subdomains').select('*').eq('subdomain_id', list_data['subdomain_id']).execute()
        if subdomain_resp.data and len(subdomain_resp.data) > 0:
            list_data['subdomain'] = subdomain_resp.data[0]
            subdomain_name = subdomain_resp.data[0]['subdomain_name']
            print(f"[DEBUG GET_LIST_DETAIL] Subdomain: {subdomain_name}")
        else:
            subdomain_name = None
            print(f"[DEBUG GET_LIST_DETAIL] No subdomain found")
        
        # Get latest version info if exists (filter by is_current=True)
        version_resp = sb.table('list_versions').select('*').eq('request_id', list_id).eq('is_current', True).order('version_number', desc=True).limit(1).execute()
        current_version = None
        if version_resp.data and len(version_resp.data) > 0:
            current_version = version_resp.data[0]
            list_data['current_version'] = current_version
            print(f"[DEBUG GET_LIST_DETAIL] Current version: {current_version['version_number']}, version_id: {current_version['version_id']}")
            
            # Get items for the current version from the appropriate entry table
            if subdomain_name:
                table_mapping = {
                    'Target Lists': 'target_list_entries',
                    'Call Lists': 'call_list_entries',
                    'Formulary Decision-Maker Lists': 'formulary_decision_maker_entries',
                    'IDN/Health System Lists': 'idn_health_system_entries',
                    'Event Invitation Lists': 'event_invitation_entries',
                    'Digital Engagement Lists': 'digital_engagement_entries',
                    'High-Value Prescriber Lists': 'high_value_prescriber_entries',
                    'Competitor Target Lists': 'competitor_target_entries'
                }
                
                entry_table = table_mapping.get(subdomain_name)
                print(f"[DEBUG GET_LIST_DETAIL] Entry table: {entry_table}")
                
                if entry_table:
                    # Get all items for this version
                    print(f"[DEBUG GET_LIST_DETAIL] Querying {entry_table} for version_id={current_version['version_id']}")
                    items_resp = sb.table(entry_table).select('*').eq('version_id', current_version['version_id']).execute()
                    print(f"[DEBUG GET_LIST_DETAIL] Items found: {len(items_resp.data) if items_resp.data else 0}")
                    
                    if items_resp.data:
                        list_data['current_snapshot'] = {
                            'version_id': current_version['version_id'],
                            'version_number': current_version['version_number'],
                            'items': items_resp.data
                        }
                        print(f"[DEBUG GET_LIST_DETAIL] Added current_snapshot with {len(items_resp.data)} items")
                    else:
                        print(f"[DEBUG GET_LIST_DETAIL] No items found in response")
        else:
            print(f"[DEBUG GET_LIST_DETAIL] No version found for this list")
        
        print(f"[DEBUG GET_LIST_DETAIL] Returning list_data with keys: {list_data.keys()}")
        return list_data
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR GET_LIST_DETAIL] Exception: {str(e)}")
        import traceback
        print(f"[ERROR GET_LIST_DETAIL] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post('', status_code=status.HTTP_201_CREATED)
def create_list(payload: Dict[str, Any]):
    """
    Create a new list
    Required fields: subdomain_id, requester_name, request_purpose
    """
    sb = _get_supabase()
    try:
        # Validate required fields
        if 'subdomain_id' not in payload:
            raise HTTPException(status_code=400, detail='subdomain_id is required')
        if 'requester_name' not in payload:
            raise HTTPException(status_code=400, detail='requester_name is required')
        if 'request_purpose' not in payload:
            raise HTTPException(status_code=400, detail='request_purpose is required')
        
        # Set default status if not provided
        if 'status' not in payload:
            payload['status'] = 'In Progress'
        
        resp = sb.table('list_requests').insert(payload).execute()
        data = resp.data if hasattr(resp, 'data') else resp
        
        if data and len(data) > 0:
            return data[0]
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put('/{list_id}')
def update_list(list_id: int, payload: Dict[str, Any]):
    """
    Update an existing list
    """
    sb = _get_supabase()
    try:
        # Remove fields that shouldn't be updated
        payload.pop('request_id', None)
        payload.pop('created_at', None)
        
        resp = sb.table('list_requests').update(payload).eq('request_id', list_id).execute()
        data = resp.data if hasattr(resp, 'data') else resp
        
        if not data or len(data) == 0:
            raise HTTPException(status_code=404, detail='List not found')
        
        return data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete('/{list_id}')
def delete_list(list_id: int):
    """
    Delete a list
    """
    sb = _get_supabase()
    try:
        resp = sb.table('list_requests').delete().eq('request_id', list_id).execute()
        return JSONResponse(status_code=200, content={'deleted': True, 'list_id': list_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/{list_id}/items', status_code=status.HTTP_201_CREATED)
def add_items_to_list(list_id: int, payload: Dict[str, Any]):
    """
    Add items to a list (creates a new version)
    Expected payload: { "items": [...], "updated_by": "user_name" }
    """
    sb = _get_supabase()
    try:
        items = payload.get('items', [])
        updated_by = payload.get('updated_by', 'Unknown')

        print(f"[DEBUG] Adding items to list {list_id}")
        print(f"[DEBUG] Number of items: {len(items)}")

        if not items:
            raise HTTPException(status_code=400, detail='No items provided')
        
        # Get current list to find subdomain
        list_resp = sb.table('list_requests').select('*').eq('request_id', list_id).execute()
        if not list_resp.data or len(list_resp.data) == 0:
            raise HTTPException(status_code=404, detail='List not found')
        
        list_data = list_resp.data[0]
        subdomain_id = list_data['subdomain_id']
        
        # Get subdomain to determine the correct entry table
        subdomain_resp = sb.table('subdomains').select('subdomain_name').eq('subdomain_id', subdomain_id).execute()
        if not subdomain_resp.data or len(subdomain_resp.data) == 0:
            raise HTTPException(status_code=404, detail='Subdomain not found')
        
        subdomain_name = subdomain_resp.data[0]['subdomain_name']
        print(f"[DEBUG] Subdomain name: {subdomain_name}")
        
        # Map subdomain to entry table
        table_mapping = {
            'Target Lists': 'target_list_entries',
            'Call Lists': 'call_list_entries',
            'Formulary Decision-Maker Lists': 'formulary_decision_maker_entries',
            'IDN/Health System Lists': 'idn_health_system_entries',
            'Event Invitation Lists': 'event_invitation_entries',
            'Digital Engagement Lists': 'digital_engagement_entries',
            'High-Value Prescriber Lists': 'high_value_prescriber_entries',
            'Competitor Target Lists': 'competitor_target_entries'
        }
        
        entry_table = table_mapping.get(subdomain_name)
        if not entry_table:
            raise HTTPException(status_code=400, detail=f'Unknown subdomain: {subdomain_name}')
        
        print(f"[DEBUG] Using entry table: {entry_table}")

        # ✅ Get the latest version for this specific list
        version_resp = sb.table('list_versions') \
            .select('version_number') \
            .eq('request_id', list_id) \
            .order('version_number', desc=True) \
            .limit(1) \
            .execute()

        next_version = 1
        if version_resp.data and len(version_resp.data) > 0:
            next_version = version_resp.data[0]['version_number'] + 1

        print(f"[DEBUG] Creating new version: {next_version}")

        # Set all previous versions to is_current=False for this list
        sb.table('list_versions').update({'is_current': False}).eq('request_id', list_id).execute()

        # ✅ Create a new version record
        version_data = {
            'request_id': list_id,
            'version_number': next_version,
            'change_type': 'Update',
            'change_rationale': f'Added {len(items)} items via CSV upload',
            'created_by': updated_by,
            'is_current': True
        }
        version_insert = sb.table('list_versions').insert(version_data).execute()

        if not version_insert.data or len(version_insert.data) == 0:
            raise HTTPException(status_code=500, detail='Failed to create version')

        # ✅ FIX: use correct primary key name from list_versions table
        version_row = version_insert.data[0]
        version_id = version_row.get('id') or version_row.get('version_id')

        if not version_id:
            raise HTTPException(status_code=500, detail='Version ID not returned from database')

        print(f"[DEBUG] Created version_id: {version_id}")

        # ✅ Add version_id to every item
        items_with_version = [{**item, 'version_id': version_id} for item in items]

        print(f"[DEBUG] Inserting {len(items_with_version)} items into {entry_table}")

        try:
            items_resp = sb.table(entry_table).insert(items_with_version).execute()
        except Exception as insert_error:
            print(f"[ERROR] Insert failed: {insert_error}")
            import traceback
            print(traceback.format_exc())
            raise HTTPException(status_code=400, detail=str(insert_error))

        inserted_count = len(items_resp.data) if hasattr(items_resp, 'data') and items_resp.data else 0
        print(f"[DEBUG] Successfully inserted {inserted_count} items into {entry_table}")

        if inserted_count == 0:
            raise HTTPException(status_code=500, detail='No items inserted into database')

        return {
            'success': True,
            'version_id': version_id,
            'version_number': next_version,
            'items_added': inserted_count,
            'table_used': entry_table
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Exception in add_items_to_list: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))



@router.post('/{list_id}/upload-csv', status_code=status.HTTP_201_CREATED)
async def upload_csv_to_list(list_id: int, file: UploadFile = File(...), updated_by: str = "CSV Upload"):
    """
    Bulk upload CSV file for a specific list_id
    Parses CSV → creates items → reuses add_items_to_list() logic
    """
    sb = _get_supabase()
    try:
        
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail='Only CSV files are supported.')

    
        contents = await file.read()
        csv_text = contents.decode('utf-8')
        reader = csv.DictReader(StringIO(csv_text))
        rows = [dict(row) for row in reader]

        if not rows:
            raise HTTPException(status_code=400, detail='CSV file is empty.')

        
        payload = {
            "items": rows,
            "updated_by": updated_by
        }

        
        return add_items_to_list(list_id=list_id, payload=payload)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/domain/{domain_id}/worklogs', response_model=List[Dict[str, Any]])
def get_work_logs_by_domain(domain_id: int, limit: int = 100):
    """
    Get work logs for all lists in a specific domain
    Joins work_logs with list_requests and subdomains to filter by domain_id
    """
    sb = _get_supabase()
    try:
        # Query work_logs and join with list_requests and subdomains
        # to filter by domain_id
        resp = sb.table('work_logs').select(
            '*, list_requests!inner(*, subdomains!inner(*))'
        ).eq('list_requests.subdomains.domain_id', domain_id).order(
            'activity_date', desc=True
        ).limit(limit).execute()
        
        work_logs = resp.data if hasattr(resp, 'data') else resp
        
        return work_logs
    except Exception as e:
        print(f"[ERROR] Exception in get_work_logs_by_domain: {str(e)}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/domain/{domain_id}/versions', response_model=List[Dict[str, Any]])
def get_versions_by_domain(domain_id: int, limit: int = 100):
    """
    Get all list versions for a specific domain
    Joins list_versions with list_requests and subdomains to filter by domain_id
    """
    sb = _get_supabase()
    try:
        # Query list_versions and join with list_requests and subdomains
        # to filter by domain_id
        resp = sb.table('list_versions').select(
            '*, list_requests!inner(request_id, requester_name, request_purpose, status, created_at, subdomains!inner(subdomain_id, subdomain_name, domain_id))'
        ).eq('list_requests.subdomains.domain_id', domain_id).order(
            'created_at', desc=True
        ).limit(limit).execute()
        
        versions = resp.data if hasattr(resp, 'data') else resp
        
        return versions
    except Exception as e:
        print(f"[ERROR] Exception in get_versions_by_domain: {str(e)}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))