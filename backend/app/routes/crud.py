from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from app.core.database import get_supabase_client
from typing import List, Dict, Any, Optional

supabase = None
def _get_supabase():
    global supabase
    if supabase is None:
        supabase = get_supabase_client()
    return supabase

def _get_or_create_default_version(sb) -> int:
    """Get or create a default version ID for standalone entries."""
    try:
        # Try to find a default standalone version (version_number = 0)
        resp = sb.table('list_versions').select('version_id').eq('version_number', 0).limit(1).execute()
        if resp.data and len(resp.data) > 0:
            return resp.data[0]['version_id']
        
        # If no default exists, try to get any existing version
        resp = sb.table('list_versions').select('version_id').limit(1).execute()
        if resp.data and len(resp.data) > 0:
            return resp.data[0]['version_id']
        
        # If no versions exist at all, we have a problem - return None and let it fail with clear error
        return None
    except Exception as e:
        print(f"Error getting default version: {e}")
        return None

def _prepare_entry_data(item: Dict[str, Any], sb=None) -> Dict[str, Any]:
    """Prepare entry data for insertion by setting defaults for required fields."""
    # If version_id not provided, try to get a default one
    if 'version_id' not in item or item['version_id'] == '' or item['version_id'] is None:
        if sb:
            default_version_id = _get_or_create_default_version(sb)
            if default_version_id:
                item['version_id'] = default_version_id
            else:
                # Remove it and let the database error be more specific
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
        # If domain_id is provided, join with subdomains to filter
        if domain_id is not None:
            query = sb.table('list_requests').select('*, subdomains(*)')
            resp = query.limit(limit).execute()
            data = resp.data if hasattr(resp, 'data') else resp
            # Filter by domain_id from joined subdomain data
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

target_list_entries_router = APIRouter(prefix='/target_list_entries', tags=['target_list_entries'])
@target_list_entries_router.get('/', response_model=List[Dict[str, Any]])
def list_target_list_entries(limit: int = 100):
    sb = _get_supabase()
    try:
        resp = sb.table('target_list_entries').select('*').limit(limit).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@target_list_entries_router.post('/', status_code=status.HTTP_201_CREATED)
def create_target_list_entries(item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        item = _prepare_entry_data(item, sb)
        resp = sb.table('target_list_entries').insert(item).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@target_list_entries_router.get('/{item_id}')
def get_target_list_entries(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('target_list_entries').select('*').eq('entry_id', item_id).execute()
        data = resp.data if hasattr(resp, 'data') else resp
        if not data:
            raise HTTPException(status_code=404, detail='Not found')
        return data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@target_list_entries_router.put('/{item_id}')
def update_target_list_entries(item_id: int, item: Dict[str, Any]):
    sb = _get_supabase()
    try:
        resp = sb.table('target_list_entries').update(item).eq('entry_id', item_id).execute()
        return resp.data if hasattr(resp, 'data') else resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@target_list_entries_router.delete('/{item_id}')
def delete_target_list_entries(item_id: int):
    sb = _get_supabase()
    try:
        resp = sb.table('target_list_entries').delete().eq('entry_id', item_id).execute()
        return JSONResponse(status_code=200, content={'deleted': True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

routers.append(target_list_entries_router)

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
