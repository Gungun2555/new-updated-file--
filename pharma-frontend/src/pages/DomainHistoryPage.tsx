import React, { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { getWorkLogsByDomain } from '../api/listApi'
import { DOMAIN_CONFIGS } from '../constants/domains'
import type { WorkLog } from '../types'

export default function DomainHistoryPage() {
  const { domainKey } = useParams()
  const navigate = useNavigate()
  const [workLogs, setWorkLogs] = useState<WorkLog[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

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
        const logs = await getWorkLogsByDomain(domainConfig.domainId, 200)
        setWorkLogs(logs)
      } catch (err) {
        console.error('Failed to fetch work logs:', err)
        setError('Failed to load work logs')
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
          <p className="text-slate-600 font-medium">Loading work logs...</p>
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
              <h1 className="text-2xl font-bold text-slate-800">Domain History</h1>
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
            <p className="text-slate-500 mb-6">Unable to load work logs for this domain</p>
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
                {domainConfig.displayName} - Work History
              </h1>
              <p className="text-sm text-slate-500 mt-1">
                Activity logs for all lists in this domain
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
        {/* Stats Card */}
        <div className="mb-6 bg-white rounded-2xl p-6 border border-slate-200 shadow-sm">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <div>
              <div className="text-3xl font-bold text-slate-800">{workLogs.length}</div>
              <div className="text-sm text-slate-500">Total Activity Logs</div>
            </div>
          </div>
        </div>

        {/* Work Logs List */}
        {workLogs.length === 0 ? (
          <div className="text-center py-16 bg-white rounded-2xl border border-slate-200">
            <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-slate-100 flex items-center justify-center">
              <svg className="w-10 h-10 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-slate-800 mb-2">No work logs yet</h3>
            <p className="text-slate-500">There are no activity logs for this domain</p>
          </div>
        ) : (
          <div className="space-y-4">
            {workLogs.map((log, index) => (
              <div
                key={log.log_id || index}
                className="bg-white rounded-xl p-6 border border-slate-200 hover:border-primary/30 hover:shadow-lg transition-all duration-300"
              >
                <div className="flex items-start gap-4">
                  {/* Timeline dot */}
                  <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-primary/10 to-secondary/10 flex items-center justify-center border border-primary/20">
                    <div className="w-3 h-3 rounded-full bg-gradient-to-br from-primary to-secondary"></div>
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    {/* Header */}
                    <div className="flex items-start justify-between gap-4 mb-3">
                      <div className="flex-1">
                        <h3 className="text-lg font-semibold text-slate-800 mb-1">
                          {log.activity_description}
                        </h3>
                        {log.list_requests && (
                          <div className="flex items-center gap-2 text-sm text-slate-600">
                            <span className="font-medium">Request Purpose:</span>
                            <span className="px-2 py-1 bg-slate-100 rounded-lg">
                              {log.list_requests.request_purpose}
                            </span>
                            {log.list_requests.subdomains && (
                              <>
                                <span className="text-slate-400">•</span>
                                <span className="px-2 py-1 bg-blue-50 text-blue-700 rounded-lg text-xs">
                                  {log.list_requests.subdomains.subdomain_name}
                                </span>
                              </>
                            )}
                          </div>
                        )}
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-medium text-slate-700">
                          {log.worker_name}
                        </div>
                        <div className="text-xs text-slate-500 mt-1">
                          {formatDate(log.activity_date)}
                        </div>
                      </div>
                    </div>

                    {/* Decisions Made */}
                    {log.decisions_made && (
                      <div className="mt-3 p-3 bg-slate-50 rounded-lg border border-slate-200">
                        <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">
                          Decisions Made
                        </div>
                        <p className="text-sm text-slate-700">{log.decisions_made}</p>
                      </div>
                    )}

                    {/* Metadata */}
                    <div className="mt-3 flex items-center gap-4 text-xs text-slate-500">
                      {log.list_requests?.requester_name && (
                        <div className="flex items-center gap-1">
                          <span className="font-medium">Requester:</span>
                          <span className="px-2 py-0.5 bg-purple-50 text-purple-700 rounded font-medium">
                            {log.list_requests.requester_name}
                          </span>
                          <span className="text-slate-400">•</span>
                          <span className="font-medium">ID:</span>
                          <span>{log.request_id}</span>
                        </div>
                      )}
                      {!log.list_requests?.requester_name && log.request_id && (
                        <div className="flex items-center gap-1">
                          <span className="font-medium">Request ID:</span>
                          <span>{log.request_id}</span>
                        </div>
                      )}
                      {log.version_id && (
                        <div className="flex items-center gap-1">
                          <span className="font-medium">Version ID:</span>
                          <span>{log.version_id}</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  )
}
