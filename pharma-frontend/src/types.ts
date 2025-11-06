import { DomainKey } from './constants/domains'

export type ListSummary = {
  request_id: number
  subdomain_id: number
  requester_name: string
  request_purpose: string
  status?: string
  assigned_to?: string
  created_at: string
  subdomain?: {
    subdomain_id: number
    domain_id: number
    subdomain_name: string
  }
}

export type ListDetail = ListSummary & {
  current_version?: ListVersion
  versions?: ListVersion[]
  current_snapshot?: {
    items: any[]
  }
}

export type ListVersion = {
  version_id: number
  request_id: number
  version_number: number
  changes_summary?: string
  rationale?: string
  updated_by: string
  created_at: string
}

export type WorkLog = {
  log_id?: number
  request_id: number
  version_id?: number
  worker_name: string
  activity_description: string
  decisions_made?: string
  activity_date?: string
  list_requests?: {
    request_id: number
    subdomain_id: number
    requester_name: string
    request_purpose: string
    status?: string
    subdomains?: {
      subdomain_id: number
      domain_id: number
      subdomain_name: string
    }
  }
}

export type Subdomain = {
  subdomain_id: number
  domain_id: number
  subdomain_name: string
}

export type Domain = {
  domain_id: number
  domain_name: string
  created_at?: string
}

