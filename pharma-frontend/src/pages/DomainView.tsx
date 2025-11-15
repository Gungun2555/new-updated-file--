import React, { useEffect, useState, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { getSubdomains } from '../api/listApi'
import InlineAddEntry from '../components/InlineAddEntry'
import Toast from '../components/Toast'
import ConfirmModal from '../components/ConfirmModal'
import { getDomainDisplayName, getDomainConfig } from '../constants/domains'
import axiosClient from '../api/axiosClient'

interface Subdomain {
  subdomain_id: number
  domain_id: number
  subdomain_name: string
}

interface SubdomainEntry {
  entry_id?: number
  id?: number
  [key: string]: any
}

export default function DomainView() {
  const { domainKey } = useParams()
  const navigate = useNavigate()
  const [subdomains, setSubdomains] = useState<Subdomain[]>([])
  const [selectedSubdomain, setSelectedSubdomain] = useState<Subdomain | null>(null)
  const [entries, setEntries] = useState<SubdomainEntry[]>([])
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' | 'warning' | 'info' } | null>(null)
  const [loading, setLoading] = useState(true)
  const [entriesLoading, setEntriesLoading] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [isAddingEntry, setIsAddingEntry] = useState(false)
  const [isEditMode, setIsEditMode] = useState(false)
  const [selectedEntries, setSelectedEntries] = useState<Set<number>>(new Set())
  const [isDeleting, setIsDeleting] = useState(false)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)

  const decodedDomainKey = domainKey ? decodeURIComponent(domainKey) : ''
  const domainConfig = getDomainConfig(decodedDomainKey)
  const displayDomainName = getDomainDisplayName(decodedDomainKey)

  // Map subdomain names to their corresponding entry table names
  const getEntryTableName = (subdomainName: string): string | null => {
    const tableMapping: Record<string, string> = {
      'Target Lists': 'target_list',
      'Call Lists': 'call_list_entries',
      'Formulary Decision-Maker Lists': 'formulary_decision_maker_entries',
      'IDN/Health System Lists': 'idn_health_system_entries',
      'Event Invitation Lists': 'event_invitation_entries',
      'Digital Engagement Lists': 'digital_engagement_entries',
      'High-Value Prescriber Lists': 'high_value_prescriber_entries',
      'Competitor Target Lists': 'competitor_target_entries'
    }
    return tableMapping[subdomainName] || null
  }

  // ✅ Get the ID field based on table type
  const getIdFieldName = (tableName: string): string => {
    return tableName === 'target_list' ? 'id' : 'entry_id'
  }

  // Fetch entries function wrapped in useCallback
  const fetchEntries = useCallback(async () => {
    if (!selectedSubdomain) {
      setEntries([])
      return
    }
    
    try {
      setEntriesLoading(true)
      const tableName = getEntryTableName(selectedSubdomain.subdomain_name)
      
      if (!tableName) {
        setToast({ message: `Unknown subdomain type: ${selectedSubdomain.subdomain_name}`, type: 'error' })
        setEntries([])
        return
      }

      // Fetch entries from the appropriate table
      const response = await axiosClient.get(`/api/${tableName}`)
      setEntries(response.data || [])
    } catch (error) {
      console.error('Failed to fetch entries for subdomain:', error)
      setToast({ message: 'Failed to load entries. Please try again.', type: 'error' })
      setEntries([])
    } finally {
      setEntriesLoading(false)
    }
  }, [selectedSubdomain])

  // Manual refresh function
  const handleRefresh = async () => {
    setIsRefreshing(true)
    setIsEditMode(false)
    setSelectedEntries(new Set())
    await fetchEntries()
    setTimeout(() => setIsRefreshing(false), 500)
  }

  // ✅ Get unique ID for an entry
  const getEntryId = (entry: SubdomainEntry, tableName: string): number => {
    const idField = getIdFieldName(tableName)
    return entry[idField] as number
  }

  // Toggle entry selection
  const handleToggleEntry = (entryId: number) => {
    setSelectedEntries(prev => {
      const newSet = new Set(prev)
      if (newSet.has(entryId)) {
        newSet.delete(entryId)
      } else {
        newSet.add(entryId)
      }
      return newSet
    })
  }

  // Select all entries
  const handleSelectAll = () => {
    const tableName = getEntryTableName(selectedSubdomain!.subdomain_name)
    if (!tableName) return

    if (selectedEntries.size === entries.length) {
      setSelectedEntries(new Set())
    } else {
      setSelectedEntries(new Set(entries.map(e => getEntryId(e, tableName))))
    }
  }

  // Delete selected entries
  const handleDeleteSelected = () => {
    if (selectedEntries.size === 0) {
      setToast({ message: 'No entries selected', type: 'warning' })
      return
    }
    setShowDeleteConfirm(true)
  }

  // Confirm delete action
  const confirmDelete = async () => {
    try {
      setIsDeleting(true)
      const tableName = getEntryTableName(selectedSubdomain!.subdomain_name)
      
      if (!tableName) {
        setToast({ message: 'Failed to delete entries', type: 'error' })
        setShowDeleteConfirm(false)
        return
      }

      // Delete each selected entry
      const deletePromises = Array.from(selectedEntries).map(entryId =>
        axiosClient.delete(`/api/${tableName}/${entryId}`)
      )

      await Promise.all(deletePromises)

      setToast({ message: `Successfully deleted ${selectedEntries.size} ${selectedEntries.size === 1 ? 'entry' : 'entries'}`, type: 'success' })
      setSelectedEntries(new Set())
      setIsEditMode(false)
      setShowDeleteConfirm(false)
      await fetchEntries()
    } catch (error) {
      console.error('Failed to delete entries:', error)
      setToast({ message: 'Failed to delete some entries. Please try again.', type: 'error' })
      setShowDeleteConfirm(false)
    } finally {
      setIsDeleting(false)
    }
  }

  // Fetch subdomains for the selected domain
  useEffect(() => {
    (async () => {
      if (!domainConfig) return
      
      try {
        setLoading(true)
        setIsEditMode(false)
        setSelectedEntries(new Set())
        
        const fetchedSubdomains = await getSubdomains(domainConfig.domainId)
        setSubdomains(fetchedSubdomains)

        // Auto-select first subdomain if available
        if (fetchedSubdomains.length > 0) {
          setSelectedSubdomain(fetchedSubdomains[0])
        }
      } catch (error) {
        console.error('Failed to fetch subdomains:', error)
        setToast({ message: 'Failed to load subdomains. Please try again.', type: 'error' })
      } finally {
        setLoading(false)
      }
    })()
  }, [decodedDomainKey, domainConfig])

  // Fetch entries when subdomain is selected
  useEffect(() => {
    setIsEditMode(false)
    setSelectedEntries(new Set())
    void fetchEntries()
  }, [fetchEntries])

  // Auto-refresh polling every 2 minutes (but pause when user is adding entry)
  useEffect(() => {
    if (isAddingEntry) {
      return
    }

    const pollInterval = setInterval(() => {
      void fetchEntries()
    }, 120000) // Poll every 2 minutes (120000 ms)

    return () => clearInterval(pollInterval)
  }, [fetchEntries, isAddingEntry])

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-slate-50 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-primary to-secondary animate-spin border-4 border-transparent border-t-white"></div>
          <p className="text-slate-600 font-medium">Loading domain...</p>
        </div>
      </div>
    )
  }

  if (!domainConfig) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-slate-50 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-100 flex items-center justify-center">
            <svg className="w-8 h-8 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h3 className="text-xl font-semibold text-slate-800 mb-2">Domain not found</h3>
          <p className="text-slate-500 mb-4">The domain you're looking for doesn't exist</p>
          <button
            onClick={() => navigate('/dashboard')}
            className="px-5 py-2.5 bg-gradient-to-r from-primary to-secondary text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    )
  }

  // Get column names for the current entry type
  const getColumnNames = (): string[] => {
    if (!entries || entries.length === 0) return []
    const firstEntry = entries[0]
    return Object.keys(firstEntry).filter(key => 
      !['entry_id', 'version_id', 'created_at', 'id'].includes(key)
    )
  }

  const columns = getColumnNames()
  const tableName = getEntryTableName(selectedSubdomain?.subdomain_name || '')
  const idFieldName = tableName ? getIdFieldName(tableName) : 'entry_id'

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-slate-50">
      {/* Header */}
      <div className="sticky top-0 z-30 backdrop-blur-xl bg-white/80 border-b border-slate-200/60 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button 
                onClick={() => navigate('/dashboard')}
                className="flex items-center gap-2 text-slate-600 hover:text-slate-900 font-medium transition-colors group"
              >
                <svg className="w-5 h-5 group-hover:-translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7" />
                </svg>
                Back to Dashboard
              </button>
              
              {/* Domain Badge */}
              <div className="px-4 py-2 bg-gradient-to-r from-primary/10 to-secondary/10 text-primary text-sm tracking-wider uppercase font-semibold rounded-lg border border-primary/20">
                {displayDomainName}
              </div>
            </div>
            
            <div className="flex gap-3">
              <button
                onClick={handleRefresh}
                disabled={isRefreshing}
                className="p-2.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-600 hover:text-slate-900 transition-all duration-200 disabled:opacity-50"
                title="Refresh entries"
              >
                <svg 
                  className={`w-5 h-5 ${isRefreshing ? 'animate-spin' : ''}`}
                  fill="none" 
                  stroke="currentColor" 
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </div>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
        {/* Subdomain Selector Section */}
        <div className="mb-6">
          <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm">
            <label className="text-sm font-semibold text-slate-700 mb-3 block">Select Subdomain</label>
            <div className="relative">
              <select
                value={selectedSubdomain?.subdomain_id || ''}
                onChange={(e) => {
                  const subdomain = subdomains.find(s => s.subdomain_id === Number(e.target.value))
                  setSelectedSubdomain(subdomain || null)
                  setIsEditMode(false)
                  setSelectedEntries(new Set())
                }}
                className="w-full h-14 pl-4 pr-12 appearance-none rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-700 font-medium focus:border-primary focus:bg-white focus:ring-4 focus:ring-primary/10 transition-all duration-200 cursor-pointer text-lg"
                disabled={subdomains.length === 0}
              >
                {subdomains.length === 0 ? (
                  <option value="">No subdomains available in this domain</option>
                ) : (
                  subdomains.map((subdomain) => (
                    <option key={subdomain.subdomain_id} value={subdomain.subdomain_id}>
                      {subdomain.subdomain_name}
                    </option>
                  ))
                )}
              </select>
              <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none">
                <svg className="w-6 h-6 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                </svg>
              </div>
            </div>
            {subdomains.length > 0 && (
              <p className="mt-3 text-sm text-slate-500">
                {subdomains.length} subdomain{subdomains.length !== 1 ? 's' : ''} available in {displayDomainName}
              </p>
            )}
          </div>
        </div>

        {/* Entries Table */}
        {selectedSubdomain && (
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
            <div className="p-6 border-b border-slate-200">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold text-slate-800">{selectedSubdomain.subdomain_name}</h2>
                  <p className="text-slate-500 mt-1">
                    {entriesLoading ? 'Loading...' : `${entries.length} ${entries.length === 1 ? 'entry' : 'entries'}`}
                    {isEditMode && selectedEntries.size > 0 && (
                      <span className="ml-2 text-primary font-semibold">({selectedEntries.size} selected)</span>
                    )}
                  </p>
                </div>
                {/* Action Buttons - Top Right */}
                <div className="flex items-center gap-2">
                  {isEditMode ? (
                    <>
                      <button
                        onClick={() => {
                          setIsEditMode(false)
                          setSelectedEntries(new Set())
                        }}
                        className="px-4 py-2.5 bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold rounded-xl transition-all"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={handleDeleteSelected}
                        disabled={selectedEntries.size === 0 || isDeleting}
                        className="px-4 py-2.5 bg-red-500 hover:bg-red-600 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl hover:shadow-red-500/30 transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center gap-2"
                      >
                        {isDeleting ? (
                          <>
                            <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            Deleting...
                          </>
                        ) : (
                          <>
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>
                            Delete Selected
                          </>
                        )}
                      </button>
                    </>
                  ) : (
                    <>
                      <button
                        onClick={() => setIsEditMode(true)}
                        className="px-5 py-2.5 bg-slate-600 hover:bg-slate-700 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 flex items-center gap-2"
                      >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                        </svg>
                        Edit
                      </button>
                      <InlineAddEntry
                        columns={columns}
                        tableName={getEntryTableName(selectedSubdomain.subdomain_name) || ''}
                        onEntryAdded={fetchEntries}
                        onAddingStateChange={setIsAddingEntry}
                        showAsButton={true}
                        subdomainName={selectedSubdomain.subdomain_name}
                      />
                    </>
                  )}
                </div>
              </div>
            </div>

            {entriesLoading ? (
              <div className="p-12 text-center">
                <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-gradient-to-br from-primary to-secondary animate-spin border-4 border-transparent border-t-white"></div>
                <p className="text-slate-600">Loading entries...</p>
              </div>
            ) : (
              <>
                {entries.length === 0 ? (
                  <div className="p-12 text-center">
                    <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-slate-100 flex items-center justify-center">
                      <svg className="w-8 h-8 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
                      </svg>
                    </div>
                    <h3 className="text-lg font-semibold text-slate-800 mb-2">No entries yet</h3>
                    <p className="text-slate-500 mb-4">Click the button below to add your first entry</p>
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead className="bg-slate-50 border-b border-slate-200">
                        <tr>
                          {isEditMode && (
                            <th className="px-4 py-4 text-left w-12">
                              <input
                                type="checkbox"
                                checked={entries.length > 0 && selectedEntries.size === entries.length}
                                onChange={handleSelectAll}
                                className="w-5 h-5 text-primary border-slate-300 rounded focus:ring-2 focus:ring-primary cursor-pointer"
                              />
                            </th>
                          )}
                          {columns.map((column) => (
                            <th key={column} className="px-6 py-4 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">
                              {column.replace(/_/g, ' ')}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {entries.map((entry) => {
                          const entryId = getEntryId(entry, tableName || '')
                          return (
                            <tr key={entryId} className="hover:bg-slate-50 transition-colors">
                              {isEditMode && (
                                <td className="px-4 py-4 w-12">
                                  <input
                                    type="checkbox"
                                    checked={selectedEntries.has(entryId)}
                                    onChange={() => handleToggleEntry(entryId)}
                                    className="w-5 h-5 text-primary border-slate-300 rounded focus:ring-2 focus:ring-primary cursor-pointer"
                                  />
                                </td>
                              )}
                              {columns.map((column) => (
                                <td key={column} className="px-6 py-4 text-sm text-slate-700">
                                  {entry[column] !== null && entry[column] !== undefined ? String(entry[column]) : '-'}
                                </td>
                              ))}
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </>
            )}
            
            {/* Form appears here when adding - only show form, not button */}
            {isAddingEntry && selectedSubdomain && (
              <InlineAddEntry
                columns={columns}
                tableName={getEntryTableName(selectedSubdomain.subdomain_name) || ''}
                onEntryAdded={fetchEntries}
                onAddingStateChange={setIsAddingEntry}
                showAsButton={false}
                subdomainName={selectedSubdomain.subdomain_name}
              />
            )}
          </div>
        )}
      </main>

      {/* Toast Notification */}
      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <ConfirmModal
          title="Delete Entries"
          message={`Are you sure you want to delete ${selectedEntries.size} ${selectedEntries.size === 1 ? 'entry' : 'entries'}? This action cannot be undone.`}
          confirmText="Delete"
          cancelText="Cancel"
          confirmColor="red"
          onConfirm={confirmDelete}
          onCancel={() => setShowDeleteConfirm(false)}
          isLoading={isDeleting}
        />
      )}
    </div>
  )
}