from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from app.core.database import get_supabase_client
from typing import List, Dict, Any, Optional
from datetime import datetime

supabase = None
def _get_supabase():
    global supabase
    if supabase is None:
        supabase = get_supabase_client()
    return supabase

# ✅ Count total rows in any table
def count_total_rows(sb, table_name: str) -> int:
    """Return total number of rows for a given table."""
    try:
        resp = sb.table(table_name).select("*", count="exact").execute()
        return resp.count or 0
    except Exception as e:
        print(f"❌ Error counting rows in {table_name}: {e}")
        return 0


# ✅ Count total change entries for a table (based on list_versions or global_version_control)
def count_total_changes(sb, table_name: str) -> int:
    """Return number of version changes logged for a given table."""
    try:
        resp = (
            sb.table("global_version_control")
            .select("id", count="exact")
            .eq("table_name", table_name)
            .execute()
        )
        return resp.count or 0
    except Exception as e:
        print(f"❌ Error counting version changes for {table_name}: {e}")
        return 0


def update_global_version(sb, table_name: str, change_type: str, changed_by: str = "system"):
    """
    Insert entry in global_version_control dynamically when data changes.
    Keeps total rows and total changes up to date.
    """
    try:
        total_rows = count_total_rows(sb, table_name)
        total_changes = count_total_changes(sb, table_name) + 1
        version_number = total_changes

        record = {
            "table_name": table_name,
            "version_number": version_number,
            "change_type": change_type,
            "changed_rows": 1,
            "total_rows": total_rows,
            "change_summary": f"{change_type} operation on {table_name}",
            "triggered_by": changed_by,
            "timestamp": datetime.utcnow().isoformat(),
        }

        sb.table("global_version_control").insert(record).execute()
        print(f"✅ Global version updated for {table_name} → {change_type}")
    except Exception as e:
        print(f"❌ Error updating global_version_control for {table_name}: {e}")


def _get_or_create_default_version(sb) -> int:
    """Get or create a default version ID for standalone entries."""
    try:
        resp = sb.table('list_versions').select('version_id').eq('version_number', 0).limit(1).execute()
        if resp.data and len(resp.data) > 0:
            return resp.data[0]['version_id']
        
        resp = sb.table('list_versions').select('version_id').limit(1).execute()
        if resp.data and len(resp.data) > 0:
            return resp.data[0]['version_id']
        
        default_version = {
            "version_number": 0,
            "change_type": "DEFAULT",
            "change_rationale": "Default version for standalone entries",
            "created_by": "system",
            "is_current": True
        }
        resp = sb.table('list_versions').insert(default_version).execute()
        if resp.data and len(resp.data) > 0:
            return resp.data[0]['version_id']
        
        return None
    except Exception as e:
        print(f"Error getting or creating default version: {e}")
        return None

def record_version_change(sb, table_name: str, change_type: str, triggered_by: str = "system", change_summary: str = "", request_id: int = None):
    """
    Record a version entry in both list_versions and global_version_control
    whenever an insert/update/delete occurs.
    """
    try:
        resp = sb.table('global_version_control').select('version_number').eq('table_name', table_name).order('version_number', desc=True).limit(1).execute()
        last_version = resp.data[0]['version_number'] if resp.data else 0
        new_version = last_version + 1

        if request_id:
            version_payload = {
                "request_id": request_id,
                "version_number": new_version,
                "change_type": change_type,
                "change_rationale": change_summary or f"{change_type} operation on {table_name}",
                "created_by": triggered_by,
                "is_current": True
            }
            sb.table('list_versions').insert(version_payload).execute()

        global_payload = {
            "table_name": table_name,
            "version_number": new_version,
            "change_type": change_type,
            "changed_rows": 1,
            "change_summary": change_summary or f"{change_type} operation in {table_name}",
            "triggered_by": triggered_by
        }
        sb.table('global_version_control').insert(global_payload).execute()

    except Exception as e:
        print(f"[VersionControlError] Failed to record version change: {e}")


def _prepare_entry_data(item: Dict[str, Any], sb=None) -> Dict[str, Any]:
    """Prepare entry data for insertion by setting defaults for required fields."""
    if 'version_id' not in item or item['version_id'] == '' or item['version_id'] is None:
        if sb:
            default_version_id = _get_or_create_default_version(sb)
            if default_version_id:
                item['version_id'] = default_version_id
            else:
                item.pop('version_id', None)
        else:
            item.pop('version_id', None)
    return item

routers = []

call_list_entries_router = APIRouter(prefix='/call_list_entries', tags=['call_list_entries'])
@call_list_entries_router.get('/', response_model=List[Dict[str, Any]])
def list_call_list_entries(limit: int = 100):
    sb = _get_supabase()
    try:
        resp = sb.table('call_list_entries').select('*').limit(limit).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@call_list_entries_router.post('/', status_code=status.HTTP_201_CREATED)
def create_call_list_entries(item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        item = _prepare_entry_data(item, sb)
        resp = sb.table('call_list_entries').insert(item).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@call_list_entries_router.get('/{item_id}')
def get_call_list_entries(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('call_list_entries').select('*').eq('entry_id', item_id).execute()
        data = resp.data if hasattr(resp, 'data') else resp
        if not data:
            raise HTTPException(status_code=404, detail='Not found')
        return data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@call_list_entries_router.put('/{item_id}')
def update_call_list_entries(item_id: int, item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        resp = sb.table('call_list_entries').update(item).eq('entry_id', item_id).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@call_list_entries_router.delete('/{item_id}')
def delete_call_list_entries(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('call_list_entries').delete().eq('entry_id', item_id).execute()
        return JSONResponse(status_code=200, content={'deleted': True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

routers.append(call_list_entries_router)

competitor_target_entries_router = APIRouter(prefix='/competitor_target_entries', tags=['competitor_target_entries'])
@competitor_target_entries_router.get('/', response_model=List[Dict[str, Any]])
def list_competitor_target_entries(limit: int = 100):
    sb = _get_supabase()
    try:
        resp = sb.table('competitor_target_entries').select('*').limit(limit).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@competitor_target_entries_router.post('/', status_code=status.HTTP_201_CREATED)
def create_competitor_target_entries(item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        item = _prepare_entry_data(item, sb)
        resp = sb.table('competitor_target_entries').insert(item).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@competitor_target_entries_router.get('/{item_id}')
def get_competitor_target_entries(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('competitor_target_entries').select('*').eq('entry_id', item_id).execute()
        data = resp.data if hasattr(resp, 'data') else resp
        if not data:
            raise HTTPException(status_code=404, detail='Not found')
        return data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@competitor_target_entries_router.put('/{item_id}')
def update_competitor_target_entries(item_id: int, item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        resp = sb.table('competitor_target_entries').update(item).eq('entry_id', item_id).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@competitor_target_entries_router.delete('/{item_id}')
def delete_competitor_target_entries(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('competitor_target_entries').delete().eq('entry_id', item_id).execute()
        return JSONResponse(status_code=200, content={'deleted': True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

routers.append(competitor_target_entries_router)

digital_engagement_entries_router = APIRouter(prefix='/digital_engagement_entries', tags=['digital_engagement_entries'])
@digital_engagement_entries_router.get('/', response_model=List[Dict[str, Any]])
def list_digital_engagement_entries(limit: int = 100):
    sb = _get_supabase()
    try:
        resp = sb.table('digital_engagement_entries').select('*').limit(limit).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@digital_engagement_entries_router.post('/', status_code=status.HTTP_201_CREATED)
def create_digital_engagement_entries(item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        item = _prepare_entry_data(item, sb)
        resp = sb.table('digital_engagement_entries').insert(item).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@digital_engagement_entries_router.get('/{item_id}')
def get_digital_engagement_entries(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('digital_engagement_entries').select('*').eq('entry_id', item_id).execute()
        data = resp.data if hasattr(resp, 'data') else resp
        if not data:
            raise HTTPException(status_code=404, detail='Not found')
        return data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@digital_engagement_entries_router.put('/{item_id}')
def update_digital_engagement_entries(item_id: int, item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        resp = sb.table('digital_engagement_entries').update(item).eq('entry_id', item_id).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@digital_engagement_entries_router.delete('/{item_id}')
def delete_digital_engagement_entries(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('digital_engagement_entries').delete().eq('entry_id', item_id).execute()
        return JSONResponse(status_code=200, content={'deleted': True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

routers.append(digital_engagement_entries_router)

domains_router = APIRouter(prefix='/domains', tags=['domains'])
@domains_router.get('/', response_model=List[Dict[str, Any]])
def list_domains(limit: int = 100):
    sb = _get_supabase()
    try:
        resp = sb.table('domains').select('*').limit(limit).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@domains_router.post('/', status_code=status.HTTP_201_CREATED)
def create_domains(item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        resp = sb.table('domains').insert(item).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@domains_router.get('/{item_id}')
def get_domains(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('domains').select('*').eq('domain_id', item_id).execute()
        data = resp.data if hasattr(resp, 'data') else resp
        if not data:
            raise HTTPException(status_code=404, detail='Not found')
        return data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@domains_router.put('/{item_id}')
def update_domains(item_id: int, item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        resp = sb.table('domains').update(item).eq('domain_id', item_id).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@domains_router.delete('/{item_id}')
def delete_domains(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('domains').delete().eq('domain_id', item_id).execute()
        return JSONResponse(status_code=200, content={'deleted': True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

routers.append(domains_router)

event_invitation_entries_router = APIRouter(prefix='/event_invitation_entries', tags=['event_invitation_entries'])
@event_invitation_entries_router.get('/', response_model=List[Dict[str, Any]])
def list_event_invitation_entries(limit: int = 100):
    sb = _get_supabase()
    try:
        resp = sb.table('event_invitation_entries').select('*').limit(limit).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@event_invitation_entries_router.post('/', status_code=status.HTTP_201_CREATED)
def create_event_invitation_entries(item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        item = _prepare_entry_data(item, sb)
        resp = sb.table('event_invitation_entries').insert(item).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@event_invitation_entries_router.get('/{item_id}')
def get_event_invitation_entries(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('event_invitation_entries').select('*').eq('entry_id', item_id).execute()
        data = resp.data if hasattr(resp, 'data') else resp
        if not data:
            raise HTTPException(status_code=404, detail='Not found')
        return data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@event_invitation_entries_router.put('/{item_id}')
def update_event_invitation_entries(item_id: int, item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        resp = sb.table('event_invitation_entries').update(item).eq('entry_id', item_id).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@event_invitation_entries_router.delete('/{item_id}')
def delete_event_invitation_entries(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('event_invitation_entries').delete().eq('entry_id', item_id).execute()
        return JSONResponse(status_code=200, content={'deleted': True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

routers.append(event_invitation_entries_router)

formulary_decision_maker_entries_router = APIRouter(prefix='/formulary_decision_maker_entries', tags=['formulary_decision_maker_entries'])
@formulary_decision_maker_entries_router.get('/', response_model=List[Dict[str, Any]])
def list_formulary_decision_maker_entries(limit: int = 100):
    sb = _get_supabase()
    try:
        resp = sb.table('formulary_decision_maker_entries').select('*').limit(limit).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@formulary_decision_maker_entries_router.post('/', status_code=status.HTTP_201_CREATED)
def create_formulary_decision_maker_entries(item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        item = _prepare_entry_data(item, sb)
        resp = sb.table('formulary_decision_maker_entries').insert(item).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@formulary_decision_maker_entries_router.get('/{item_id}')
def get_formulary_decision_maker_entries(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('formulary_decision_maker_entries').select('*').eq('entry_id', item_id).execute()
        data = resp.data if hasattr(resp, 'data') else resp
        if not data:
            raise HTTPException(status_code=404, detail='Not found')
        return data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@formulary_decision_maker_entries_router.put('/{item_id}')
def update_formulary_decision_maker_entries(item_id: int, item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        resp = sb.table('formulary_decision_maker_entries').update(item).eq('entry_id', item_id).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@formulary_decision_maker_entries_router.delete('/{item_id}')
def delete_formulary_decision_maker_entries(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('formulary_decision_maker_entries').delete().eq('entry_id', item_id).execute()
        return JSONResponse(status_code=200, content={'deleted': True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

routers.append(formulary_decision_maker_entries_router)

high_value_prescriber_entries_router = APIRouter(prefix='/high_value_prescriber_entries', tags=['high_value_prescriber_entries'])
@high_value_prescriber_entries_router.get('/', response_model=List[Dict[str, Any]])
def list_high_value_prescriber_entries(limit: int = 100):
    sb = _get_supabase()
    try:
        resp = sb.table('high_value_prescriber_entries').select('*').limit(limit).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@high_value_prescriber_entries_router.post('/', status_code=status.HTTP_201_CREATED)
def create_high_value_prescriber_entries(item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        item = _prepare_entry_data(item, sb)
        resp = sb.table('high_value_prescriber_entries').insert(item).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@high_value_prescriber_entries_router.get('/{item_id}')
def get_high_value_prescriber_entries(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('high_value_prescriber_entries').select('*').eq('entry_id', item_id).execute()
        data = resp.data if hasattr(resp, 'data') else resp
        if not data:
            raise HTTPException(status_code=404, detail='Not found')
        return data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@high_value_prescriber_entries_router.put('/{item_id}')
def update_high_value_prescriber_entries(item_id: int, item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        resp = sb.table('high_value_prescriber_entries').update(item).eq('entry_id', item_id).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@high_value_prescriber_entries_router.delete('/{item_id}')
def delete_high_value_prescriber_entries(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('high_value_prescriber_entries').delete().eq('entry_id', item_id).execute()
        return JSONResponse(status_code=200, content={'deleted': True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

routers.append(high_value_prescriber_entries_router)

idn_health_system_entries_router = APIRouter(prefix='/idn_health_system_entries', tags=['idn_health_system_entries'])
@idn_health_system_entries_router.get('/', response_model=List[Dict[str, Any]])
def list_idn_health_system_entries(limit: int = 100):
    sb = _get_supabase()
    try:
        resp = sb.table('idn_health_system_entries').select('*').limit(limit).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@idn_health_system_entries_router.post('/', status_code=status.HTTP_201_CREATED)
def create_idn_health_system_entries(item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        item = _prepare_entry_data(item, sb)
        resp = sb.table('idn_health_system_entries').insert(item).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@idn_health_system_entries_router.get('/{item_id}')
def get_idn_health_system_entries(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('idn_health_system_entries').select('*').eq('entry_id', item_id).execute()
        data = resp.data if hasattr(resp, 'data') else resp
        if not data:
            raise HTTPException(status_code=404, detail='Not found')
        return data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@idn_health_system_entries_router.put('/{item_id}')
def update_idn_health_system_entries(item_id: int, item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        resp = sb.table('idn_health_system_entries').update(item).eq('entry_id', item_id).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@idn_health_system_entries_router.delete('/{item_id}')
def delete_idn_health_system_entries(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('idn_health_system_entries').delete().eq('entry_id', item_id).execute()
        return JSONResponse(status_code=200, content={'deleted': True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

routers.append(idn_health_system_entries_router)

list_requests_router = APIRouter(prefix='/list_requests', tags=['list_requests'])
@list_requests_router.get('/', response_model=List[Dict[str, Any]])
def list_list_requests(limit: int = 100, subdomain_id: Optional[int] = None, domain_id: Optional[int] = None):
    sb = _get_supabase()
    try:
        if domain_id is not None:
            query = sb.table('list_requests').select(', subdomains()')
            resp = query.limit(limit).execute()
            data = resp.data if hasattr(resp, 'data') else resp
            filtered_data = [item for item in data if item.get('subdomains') and item['subdomains'].get('domain_id') == domain_id]
            return filtered_data
        else:
            query = sb.table('list_requests').select('*')
            if subdomain_id is not None:
                query = query.eq('subdomain_id', subdomain_id)
            resp = query.limit(limit).execute()
            return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@list_requests_router.post('/', status_code=status.HTTP_201_CREATED)
def create_list_requests(item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        resp = sb.table('list_requests').insert(item).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@list_requests_router.get('/{item_id}')
def get_list_requests(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('list_requests').select('*').eq('request_id', item_id).execute()
        data = resp.data if hasattr(resp, 'data') else resp
        if not data:
            raise HTTPException(status_code=404, detail='Not found')
        return data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@list_requests_router.put('/{item_id}')
def update_list_requests(item_id: int, item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        resp = sb.table('list_requests').update(item).eq('request_id', item_id).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@list_requests_router.delete('/{item_id}')
def delete_list_requests(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('list_requests').delete().eq('request_id', item_id).execute()
        return JSONResponse(status_code=200, content={'deleted': True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

routers.append(list_requests_router)

list_versions_router = APIRouter(prefix='/list_versions', tags=['list_versions'])
@list_versions_router.get('/', response_model=List[Dict[str, Any]])
def list_list_versions(limit: int = 100):
    sb = _get_supabase()
    try:
        resp = sb.table('list_versions').select('*').limit(limit).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@list_versions_router.post('/', status_code=status.HTTP_201_CREATED)
def create_list_versions(item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        resp = sb.table('list_versions').insert(item).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@list_versions_router.get('/{item_id}')
def get_list_versions(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('list_versions').select('*').eq('version_id', item_id).execute()
        data = resp.data if hasattr(resp, 'data') else resp
        if not data:
            raise HTTPException(status_code=404, detail='Not found')
        return data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@list_versions_router.put('/{item_id}')
def update_list_versions(item_id: int, item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        resp = sb.table('list_versions').update(item).eq('version_id', item_id).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@list_versions_router.delete('/{item_id}')
def delete_list_versions(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('list_versions').delete().eq('version_id', item_id).execute()
        return JSONResponse(status_code=200, content={'deleted': True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

routers.append(list_versions_router)

subdomains_router = APIRouter(prefix='/subdomains', tags=['subdomains'])
@subdomains_router.get('/', response_model=List[Dict[str, Any]])
def list_subdomains(limit: int = 100, domain_id: Optional[int] = None):
    sb = _get_supabase()
    try:
        query = sb.table('subdomains').select('*')
        if domain_id is not None:
            query = query.eq('domain_id', domain_id)
        resp = query.limit(limit).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@subdomains_router.post('/', status_code=status.HTTP_201_CREATED)
def create_subdomains(item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        resp = sb.table('subdomains').insert(item).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@subdomains_router.get('/{item_id}')
def get_subdomains(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('subdomains').select('*').eq('subdomain_id', item_id).execute()
        data = resp.data if hasattr(resp, 'data') else resp
        if not data:
            raise HTTPException(status_code=404, detail='Not found')
        return data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@subdomains_router.put('/{item_id}')
def update_subdomains(item_id: int, item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        resp = sb.table('subdomains').update(item).eq('subdomain_id', item_id).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@subdomains_router.delete('/{item_id}')
def delete_subdomains(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('subdomains').delete().eq('subdomain_id', item_id).execute()
        return JSONResponse(status_code=200, content={'deleted': True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

routers.append(subdomains_router)

# ============================================
# TARGET LIST CRUD - NEW SCHEMA
# ============================================
# Table: target_list (standalone, no version_id)
# Version control via database triggers
# ============================================

target_list_router = APIRouter(prefix='/target_list', tags=['target_list'])

@target_list_router.get('/', response_model=List[Dict[str, Any]])
def list_target_list(limit: int = 100):
    """Get all target list entries"""
    sb = _get_supabase()
    try:
        resp = sb.table('target_list').select('*').limit(limit).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@target_list_router.post('/', status_code=status.HTTP_201_CREATED)
def create_target_list_entry(item: Dict[str, Any]):
    """
    Create a new target list entry
    Required fields: full_name
    Database triggers automatically log to history_table
    """
    sb = _get_supabase()
    try:
        if 'full_name' not in item or not item['full_name']:
            raise HTTPException(status_code=400, detail='full_name is required')
        
        valid_columns = {
            'hcp_code', 'full_name', 'gender', 'qualification', 'specialty',
            'designation', 'email', 'phone', 'hospital_name', 'hospital_address',
            'city', 'state', 'pincode', 'experience_years', 'influence_score',
            'category', 'therapy_area', 'monthly_sales', 'yearly_sales',
            'last_interaction_date', 'call_frequency', 'priority'
        }
        
        cleaned_item = {k: v for k, v in item.items() if k in valid_columns}
        resp = sb.table('target_list').insert(cleaned_item).execute()
        
        if resp.data and len(resp.data) > 0:
            print(f"✅ Created target list entry: {resp.data[0].get('id')}")
            return resp.data[0]
        
        raise HTTPException(status_code=500, detail='Failed to create entry')
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error creating target list entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@target_list_router.get('/{item_id}')
def get_target_list_entry(item_id: int):
    """Get a specific target list entry by ID"""
    sb = _get_supabase()
    try:
        resp = sb.table('target_list').select('*').eq('id', item_id).execute()
        data = resp.data if hasattr(resp, 'data') else resp
        
        if not data or len(data) == 0:
            raise HTTPException(status_code=404, detail='Target list entry not found')
        
        return data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@target_list_router.put('/{item_id}')
def update_target_list_entry(item_id: int, item: Dict[str, Any]):
    """Update a target list entry"""
    sb = _get_supabase()
    try:
        valid_columns = {
            'hcp_code', 'full_name', 'gender', 'qualification', 'specialty',
            'designation', 'email', 'phone', 'hospital_name', 'hospital_address',
            'city', 'state', 'pincode', 'experience_years', 'influence_score',
            'category', 'therapy_area', 'monthly_sales', 'yearly_sales',
            'last_interaction_date', 'call_frequency', 'priority'
        }
        
        cleaned_item = {k: v for k, v in item.items() if k in valid_columns}
        
        if not cleaned_item:
            raise HTTPException(status_code=400, detail='No valid fields to update')
        
        check_resp = sb.table('target_list').select('id').eq('id', item_id).execute()
        if not check_resp.data:
            raise HTTPException(status_code=404, detail='Target list entry not found')
        
        resp = sb.table('target_list').update(cleaned_item).eq('id', item_id).execute()
        
        if resp.data and len(resp.data) > 0:
            print(f"✅ Updated target list entry: {item_id}")
            return resp.data[0]
        
        raise HTTPException(status_code=500, detail='Failed to update entry')
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error updating target list entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@target_list_router.delete('/{item_id}')
def delete_target_list_entry(item_id: int):
    """Delete a target list entry"""
    sb = _get_supabase()
    try:
        check_resp = sb.table('target_list').select('id').eq('id', item_id).execute()
        if not check_resp.data:
            raise HTTPException(status_code=404, detail='Target list entry not found')
        
        resp = sb.table('target_list').delete().eq('id', item_id).execute()
        
        print(f"✅ Deleted target list entry: {item_id}")
        
        return JSONResponse(
            status_code=200, 
            content={
                'success': True, 
                'deleted': True, 
                'id': item_id,
                'message': 'Entry deleted successfully. Version history logged automatically.'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting target list entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@target_list_router.get('/{hcp_code}/by-code')
def get_target_list_by_hcp_code(hcp_code: str):
    """Get target list entry by HCP code (unique field)"""
    sb = _get_supabase()
    try:
        resp = sb.table('target_list').select('*').eq('hcp_code', hcp_code).execute()
        data = resp.data if hasattr(resp, 'data') else resp
        
        if not data or len(data) == 0:
            raise HTTPException(status_code=404, detail=f'No entry found for HCP code: {hcp_code}')
        
        return data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@target_list_router.get('/search/by-name')
def search_target_list_by_name(name: str, limit: int = 20):
    """Search target list entries by full name (case-insensitive)"""
    sb = _get_supabase()
    try:
        resp = sb.table('target_list').select('*').ilike('full_name', f'%{name}%').limit(limit).execute()
        data = resp.data if hasattr(resp, 'data') else resp
        
        if not data:
            return []
        
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@target_list_router.get('/filter/by-specialty')
def filter_target_list_by_specialty(specialty: str, limit: int = 100):
    """Filter target list entries by specialty"""
    sb = _get_supabase()
    try:
        resp = sb.table('target_list').select('*').eq('specialty', specialty).limit(limit).execute()
        data = resp.data if hasattr(resp, 'data') else resp
        
        return data if data else []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@target_list_router.get('/filter/by-priority')
def filter_target_list_by_priority(priority: bool = True, limit: int = 100):
    """Filter target list entries by priority status"""
    sb = _get_supabase()
    try:
        resp = sb.table('target_list').select('*').eq('priority', priority).limit(limit).execute()
        data = resp.data if hasattr(resp, 'data') else resp
        
        return data if data else []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@target_list_router.get('/filter/by-category')
def filter_target_list_by_category(category: str, limit: int = 100):
    """Filter target list entries by category"""
    sb = _get_supabase()
    try:
        resp = sb.table('target_list').select('*').eq('category', category).limit(limit).execute()
        data = resp.data if hasattr(resp, 'data') else resp
        
        return data if data else []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@target_list_router.post('/bulk-create')
def bulk_create_target_list_entries(items: List[Dict[str, Any]]):
    """Bulk create multiple target list entries"""
    sb = _get_supabase()
    try:
        if not items or len(items) == 0:
            raise HTTPException(status_code=400, detail='No items provided')
        
        valid_columns = {
            'hcp_code', 'full_name', 'gender', 'qualification', 'specialty',
            'designation', 'email', 'phone', 'hospital_name', 'hospital_address',
            'city', 'state', 'pincode', 'experience_years', 'influence_score',
            'category', 'therapy_area', 'monthly_sales', 'yearly_sales',
            'last_interaction_date', 'call_frequency', 'priority'
        }
        
        cleaned_items = []
        for item in items:
            if 'full_name' not in item or not item['full_name']:
                print(f"⚠️ Skipping item without full_name: {item}")
                continue
            
            cleaned_item = {k: v for k, v in item.items() if k in valid_columns}
            cleaned_items.append(cleaned_item)
        
        if not cleaned_items:
            raise HTTPException(status_code=400, detail='No valid items to create')
        
        resp = sb.table('target_list').insert(cleaned_items).execute()
        
        inserted_count = len(resp.data) if resp.data else 0
        
        print(f"✅ Bulk created {inserted_count} target list entries")
        
        return {
            'success': True,
            'items_created': inserted_count,
            'total_provided': len(items),
            'skipped': len(items) - inserted_count,
            'message': f'Created {inserted_count} entries. All changes logged automatically via triggers.'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error bulk creating target list entries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@target_list_router.post('/bulk-update')
def bulk_update_target_list_entries(updates: List[Dict[str, Any]]):
    """Bulk update multiple target list entries"""
    sb = _get_supabase()
    try:
        if not updates or len(updates) == 0:
            raise HTTPException(status_code=400, detail='No updates provided')
        
        valid_columns = {
            'hcp_code', 'full_name', 'gender', 'qualification', 'specialty',
            'designation', 'email', 'phone', 'hospital_name', 'hospital_address',
            'city', 'state', 'pincode', 'experience_years', 'influence_score',
            'category', 'therapy_area', 'monthly_sales', 'yearly_sales',
            'last_interaction_date', 'call_frequency', 'priority'
        }
        
        updated_count = 0
        errors = []
        
        for update in updates:
            try:
                item_id = update.get('id')
                if not item_id:
                    errors.append('Missing id field in update')
                    continue
                
                update_data = {k: v for k, v in update.items() if k != 'id' and k in valid_columns}
                
                if not update_data:
                    errors.append(f'No valid fields to update for id {item_id}')
                    continue
                
                resp = sb.table('target_list').update(update_data).eq('id', item_id).execute()
                
                if resp.data:
                    updated_count += 1
                else:
                    errors.append(f'Failed to update id {item_id}')
            except Exception as e:
                errors.append(f'Error updating id {update.get("id")}: {str(e)}')
        
        return {
            'success': True,
            'items_updated': updated_count,
            'total_provided': len(updates),
            'errors': errors if errors else None,
            'message': f'Updated {updated_count} entries. All changes logged automatically via triggers.'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error bulk updating target list entries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


routers.append(target_list_router)

work_logs_router = APIRouter(prefix='/work_logs', tags=['work_logs'])
@work_logs_router.get('/', response_model=List[Dict[str, Any]])
def list_work_logs(limit: int = 100):
    sb = _get_supabase()
    try:
        resp = sb.table('work_logs').select('*').limit(limit).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@work_logs_router.post('/', status_code=status.HTTP_201_CREATED)
def create_work_logs(item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        resp = sb.table('work_logs').insert(item).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@work_logs_router.get('/{item_id}')
def get_work_logs(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('work_logs').select('*').eq('log_id', item_id).execute()
        data = resp.data if hasattr(resp, 'data') else resp
        if not data:
            raise HTTPException(status_code=404, detail='Not found')
        return data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@work_logs_router.put('/{item_id}')
def update_work_logs(item_id: int, item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        resp = sb.table('work_logs').update(item).eq('log_id', item_id).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@work_logs_router.delete('/{item_id}')
def delete_work_logs(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('work_logs').delete().eq('log_id', item_id).execute()
        return JSONResponse(status_code=200, content={'deleted': True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

routers.append(work_logs_router)

v_current_lists_router = APIRouter(prefix='/v_current_lists', tags=['v_current_lists'])
@v_current_lists_router.get('/', response_model=List[Dict[str, Any]])
def list_v_current_lists(limit: int = 100):
    sb = _get_supabase()
    try:
        resp = sb.table('v_current_lists').select('*').limit(limit).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@v_current_lists_router.post('/', status_code=status.HTTP_201_CREATED)
def create_v_current_lists(item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        resp = sb.table('v_current_lists').insert(item).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@v_current_lists_router.get('/{item_id}')
def get_v_current_lists(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('v_current_lists').select('*').eq('id', item_id).execute()
        data = resp.data if hasattr(resp, 'data') else resp
        if not data:
            raise HTTPException(status_code=404, detail='Not found')
        return data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@v_current_lists_router.put('/{item_id}')
def update_v_current_lists(item_id: int, item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        resp = sb.table('v_current_lists').update(item).eq('id', item_id).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@v_current_lists_router.delete('/{item_id}')
def delete_v_current_lists(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('v_current_lists').delete().eq('id', item_id).execute()
        return JSONResponse(status_code=200, content={'deleted': True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

routers.append(v_current_lists_router)