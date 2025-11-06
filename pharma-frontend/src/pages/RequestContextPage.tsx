import React, { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { getListRequestsByDomain, getSubdomains } from '../api/listApi'
import { DOMAIN_CONFIGS } from '../constants/domains'
import type { ListSummary, Subdomain } from '../types'

export default function RequestContextPage() {
  const { domainKey } = useParams()
  const navigate = useNavigate()
  const [listRequests, setListRequests] = useState<ListSummary[]>([])
  const [subdomains, setSubdomains] = useState<Subdomain[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedSubdomain, setSelectedSubdomain] = useState<number | null>(null)

  // Find the domain config
  const domainConfig = DOMAIN_CONFIGS.find(d => d.key === domainKey)

  useEffect(() => {
    (async () => {
      if (!domainConfig) {
        setError('Domain not found')
        setLoading(false)
        return
      }

      try {
        setLoading(true)
        setError(null)
        
        // Fetch list requests for this domain
        const requests = await getListRequestsByDomain(domainConfig.domainId, 200)
        setListRequests(requests)
        
        // Fetch subdomains for filtering
        const subdomainData = await getSubdomains(domainConfig.domainId)
        setSubdomains(subdomainData)
      } catch (err) {
        console.error('Failed to fetch list requests:', err)
        setError('Failed to load list requests')
      } finally {
        setLoading(false)
      }
    })()
  }, [domainConfig])

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-slate-50 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-primary to-secondary animate-spin border-4 border-transparent border-t-white"></div>
          <p className="text-slate-600 font-medium">Loading list requests...</p>
        </div>
      </div>
    )
  }

  if (error || !domainConfig) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-slate-50">
        <header className="sticky top-0 z-40 backdrop-blur-xl bg-white/80 border-b border-slate-200/60 shadow-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 py-5">
            <div className="flex items-center gap-4">
              <button
                onClick={() => navigate('/')}
                className="p-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-600 hover:text-slate-900 transition-all duration-200"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
              </button>
              <h1 className="text-2xl font-bold text-slate-800">Request Context</h1>
            </div>
          </div>
        </header>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
          <div className="text-center py-16">
            <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-red-100 flex items-center justify-center">
              <svg className="w-10 h-10 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-slate-800 mb-2">{error || 'Domain not found'}</h3>
            <p className="text-slate-500 mb-6">Unable to load list requests for this domain</p>
            <button
              onClick={() => navigate('/')}
              className="px-6 py-3 bg-gradient-to-r from-primary to-secondary text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all duration-300"
            >
              Back to Dashboard
            </button>
          </div>
        </div>
      </div>
    )
  }

  const formatDate = (dateString?: string) => {
    if (!dateString) return 'N/A'
    const date = new Date(dateString)
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  // Filter list requests by selected subdomain
  const filteredRequests = selectedSubdomain
    ? listRequests.filter(req => {
        const subdomainId = req.subdomain?.subdomain_id || (req as any).subdomains?.subdomain_id
        return subdomainId === selectedSubdomain
      })
    : listRequests

  // Group requests by subdomain
  const requestsBySubdomain = filteredRequests.reduce((acc, req) => {
    // Handle both subdomain (singular) and subdomains (plural) from backend
    const subdomainData = req.subdomain || (req as any).subdomains
    const subdomainName = subdomainData?.subdomain_name || 'Unknown'
    if (!acc[subdomainName]) {
      acc[subdomainName] = []
    }
    acc[subdomainName].push(req)
    return acc
  }, {} as Record<string, ListSummary[]>)

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-slate-50">
      {/* Header */}
      <header className="sticky top-0 z-40 backdrop-blur-xl bg-white/80 border-b border-slate-200/60 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-5">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate('/')}
              className="p-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-600 hover:text-slate-900 transition-all duration-200"
              title="Back to Dashboard"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
            </button>
            <div className="flex-1">
              <h1 className="text-2xl font-bold text-slate-800">
                {domainConfig.displayName} - Request Context
              </h1>
              <p className="text-sm text-slate-500 mt-1">
                View all list requests and their details for this domain
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
        {/* Stats and Filters */}
        <div className="mb-6 grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Stats Card */}
          <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </div>
              <div>
                <div className="text-3xl font-bold text-slate-800">{filteredRequests.length}</div>
                <div className="text-sm text-slate-500">Total List Requests</div>
              </div>
            </div>
          </div>

          {/* Subdomain Filter */}
          <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm">
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Filter by Subdomain
            </label>
            <select
              value={selectedSubdomain || ''}
              onChange={(e) => setSelectedSubdomain(e.target.value ? Number(e.target.value) : null)}
              className="w-full px-4 py-2 border border-slate-300 rounded-xl focus:ring-2 focus:ring-primary focus:border-transparent"
            >
              <option value="">All Subdomains</option>
              {subdomains.map(subdomain => (
                <option key={subdomain.subdomain_id} value={subdomain.subdomain_id}>
                  {subdomain.subdomain_name}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* List Requests */}
        {filteredRequests.length === 0 ? (
          <div className="text-center py-16 bg-white rounded-2xl border border-slate-200">
            <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-slate-100 flex items-center justify-center">
              <svg className="w-10 h-10 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-slate-800 mb-2">No list requests yet</h3>
            <p className="text-slate-500">There are no list requests for this domain</p>
          </div>
        ) : (
          <div className="space-y-6">
            {Object.entries(requestsBySubdomain).map(([subdomainName, requests]) => (
              <div key={subdomainName} className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
                {/* Subdomain Header */}
                <div className="bg-gradient-to-r from-primary/5 to-secondary/5 border-b border-slate-200 px-6 py-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center">
                        <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
                        </svg>
                      </div>
                      <div>
                        <h3 className="text-lg font-bold text-slate-800">{subdomainName}</h3>
                        <p className="text-sm text-slate-500">{requests.length} request{requests.length !== 1 ? 's' : ''}</p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Requests List */}
                <div className="divide-y divide-slate-200">
                  {requests.map((request, index) => (
                    <div
                      key={request.request_id}
                      className="p-6 hover:bg-slate-50 transition-colors duration-200 cursor-pointer"
                      onClick={() => navigate(`/list/${request.request_id}`)}
                    >
                      <div className="flex items-start justify-between gap-4">
                        {/* Left side - Request info */}
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-3 mb-2">
                            <span className="px-3 py-1 bg-primary/10 text-primary text-xs font-semibold rounded-lg">
                              ID: {request.request_id}
                            </span>
                            <span className="px-3 py-1 bg-purple-50 text-purple-700 text-xs font-semibold rounded-lg">
                              {(request.subdomain || (request as any).subdomains)?.subdomain_name || 'Unknown List'}
                            </span>
                            {request.status && (
                              <span className={`px-3 py-1 text-xs font-semibold rounded-lg ${
                                request.status === 'Completed' ? 'bg-green-100 text-green-700' :
                                request.status === 'In Progress' ? 'bg-blue-100 text-blue-700' :
                                request.status === 'Pending' ? 'bg-yellow-100 text-yellow-700' :
                                'bg-slate-100 text-slate-700'
                              }`}>
                                {request.status}
                              </span>
                            )}
                          </div>
                          <h4 className="text-lg font-semibold text-slate-800 mb-1">
                            {request.request_purpose}
                          </h4>
                          <div className="flex flex-wrap items-center gap-4 text-sm text-slate-600 mt-2">
                            <div className="flex items-center gap-1">
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                              </svg>
                              <span className="font-medium">Requester:</span>
                              <span>{request.requester_name}</span>
                            </div>
                            {request.assigned_to && (
                              <>
                                <span className="text-slate-400">•</span>
                                <div className="flex items-center gap-1">
                                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                  </svg>
                                  <span className="font-medium">Assigned to:</span>
                                  <span>{request.assigned_to}</span>
                                </div>
                              </>
                            )}
                          </div>
                        </div>

                        {/* Right side - Date and action */}
                        <div className="text-right">
                          <div className="text-sm text-slate-500 mb-2">
                            {formatDate(request.created_at)}
                          </div>
                          <button className="text-primary hover:text-secondary transition-colors text-sm font-medium">
                            View Details →
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  )
}
