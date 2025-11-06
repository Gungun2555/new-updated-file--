import axiosClient from './axiosClient'
import type { ListSummary, ListDetail, ListVersion, WorkLog } from '../types'

export async function getLists(category?: string, subdomainId?: number): Promise<ListSummary[]> {
  try {
    const params: any = {}
    if (category) params.category = category
    if (subdomainId) params.subdomain_id = subdomainId
    
    const response = await axiosClient.get('/api/lists', { 
      params,
      headers: {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
      }
    })
    return response.data
  } catch (error) {
    console.error('Error fetching lists:', error)
    throw error
  }
}

export async function getListDetail(id: string | number): Promise<ListDetail> {
  try {
    const response = await axiosClient.get(`/api/lists/${id}`)
    return response.data
  } catch (error) {
    console.error(`Error fetching list ${id}:`, error)
    throw error
  }
}

export async function createList(payload: {
  subdomain_id: number
  requester_name: string
  request_purpose: string
  status?: string
}): Promise<ListSummary> {
  try {
    const response = await axiosClient.post('/api/lists', payload)
    return response.data
  } catch (error) {
    console.error('Error creating list:', error)
    throw error
  }
}

export async function updateList(id: string | number, payload: {
  requester_name?: string
  request_purpose?: string
  status?: string
  assigned_to?: string
}): Promise<ListSummary> {
  try {
    const response = await axiosClient.put(`/api/lists/${id}`, payload)
    return response.data
  } catch (error) {
    console.error(`Error updating list ${id}:`, error)
    throw error
  }
}

export async function addItemsToList(id: string | number, items: any[], updatedBy?: string): Promise<any> {
  try {
    const response = await axiosClient.post(`/api/lists/${id}/items`, {
      items,
      updated_by: updatedBy || 'Current User'
    })
    return response.data
  } catch (error) {
    console.error(`Error adding items to list ${id}:`, error)
    throw error
  }
}

export async function deleteList(id: string | number): Promise<{ success: boolean }> {
  try {
    await axiosClient.delete(`/api/lists/${id}`)
    return { success: true }
  } catch (error) {
    console.error(`Error deleting list ${id}:`, error)
    throw error
  }
}

export async function getListVersions(listId: string): Promise<ListVersion[]> {
  try {
    const response = await axiosClient.get(`/versions/${listId}`)
    return response.data
  } catch (error) {
    console.error(`Error fetching versions for list ${listId}:`, error)
    throw error
  }
}

export async function createVersion(payload: {
  list_id: string
  version_number: number
  changes_summary: string
  rationale: string
  updated_by: string
}): Promise<ListVersion> {
  try {
    const response = await axiosClient.post('/versions', payload)
    return response.data
  } catch (error) {
    console.error('Error creating version:', error)
    throw error
  }
}

export async function getWorkLogs(listId: string): Promise<WorkLog[]> {
  try {
    const response = await axiosClient.get(`/worklogs/${listId}`)
    return response.data
  } catch (error) {
    console.error(`Error fetching work logs for list ${listId}:`, error)
    throw error
  }
}

export async function getWorkLogsByDomain(domainId: number, limit?: number): Promise<WorkLog[]> {
  try {
    const params: any = {}
    if (limit) params.limit = limit
    
    const response = await axiosClient.get(`/api/lists/domain/${domainId}/worklogs`, { params })
    return response.data
  } catch (error) {
    console.error(`Error fetching work logs for domain ${domainId}:`, error)
    throw error
  }
}

export async function getVersionsByDomain(domainId: number, limit?: number): Promise<any[]> {
  try {
    const params: any = {}
    if (limit) params.limit = limit
    
    const response = await axiosClient.get(`/api/lists/domain/${domainId}/versions`, { params })
    return response.data
  } catch (error) {
    console.error(`Error fetching versions for domain ${domainId}:`, error)
    throw error
  }
}

export async function addWorkLog(payload: {
  list_id: string
  action: string
  performed_by: string
}): Promise<WorkLog> {
  try {
    const response = await axiosClient.post('/worklogs', payload)
    return response.data
  } catch (error) {
    console.error('Error adding work log:', error)
    throw error
  }
}

export async function getSubdomains(domainId?: number): Promise<any[]> {
  try {
    const response = await axiosClient.get('/api/subdomains', {
      params: domainId ? { domain_id: domainId } : {}
    })
    return response.data
  } catch (error) {
    console.error('Error fetching subdomains:', error)
    throw error
  }
}

export async function getListsBySubdomain(subdomainId: number): Promise<any[]> {
  try {
    const response = await axiosClient.get('/api/list_requests', {
      params: { subdomain_id: subdomainId }
    })
    return response.data
  } catch (error) {
    console.error(`Error fetching lists for subdomain ${subdomainId}:`, error)
    throw error
  }
}

export async function getListRequestsByDomain(domainId: number, limit?: number): Promise<ListSummary[]> {
  try {
    const params: any = { domain_id: domainId }
    if (limit) params.limit = limit
    
    const response = await axiosClient.get('/api/list_requests', { params })
    return response.data
  } catch (error) {
    console.error(`Error fetching list requests for domain ${domainId}:`, error)
    throw error
  }
}

