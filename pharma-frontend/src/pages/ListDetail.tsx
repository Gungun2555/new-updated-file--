import React, { useEffect, useState, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { getListDetail, deleteList } from '../api/listApi'
import InlineAddEntry from '../components/InlineAddEntry'
import Toast from '../components/Toast'
import { getDomainDisplayName, migrateDomainName } from '../constants/domains'

export default function ListDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [list, setList] = useState<any>(null)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' | 'warning' | 'info' } | null>(null)
  const [isRefreshing, setIsRefreshing] = useState(false)

  // Fetch list detail function wrapped in useCallback
  const fetchListDetail = useCallback(async () => {
    if (!id) return
    try {
      const res = await getListDetail(id)
      setList(res)
    } catch (error) {
      console.error('Failed to fetch list:', error)
    }
  }, [id])

  // Initial fetch
  useEffect(() => {
    void fetchListDetail()
  }, [fetchListDetail])

  // Auto-refresh polling every 2 minutes
  useEffect(() => {
    const pollInterval = setInterval(() => {
      void fetchListDetail()
    }, 120000) // Poll every 2 minutes (120000 ms)

    return () => clearInterval(pollInterval)
  }, [fetchListDetail])

  // Manual refresh function
  const handleRefresh = async () => {
    setIsRefreshing(true)
    await fetchListDetail()
    setTimeout(() => setIsRefreshing(false), 500)
  }

  const handleDeleteList = async () => {
    if (!id) return
    try {
      await deleteList(id)
      setToast({ message: 'List deleted successfully', type: 'success' })
      setTimeout(() => navigate('/dashboard'), 1500)
    } catch (error) {
      console.error('Failed to delete list:', error)
      setToast({ message: 'Failed to delete list. Please try again.', type: 'error' })
    }
  }

  if (!list) return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-slate-50 flex items-center justify-center">
      <div className="text-center">
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-primary to-secondary animate-spin border-4 border-transparent border-t-white"></div>
        <p className="text-slate-600 font-medium">Loading list details...</p>
      </div>
    </div>
  )

  // Map backend fields to display fields
  const displayTitle = list.subdomain?.subdomain_name || list.request_purpose || list.purpose || list.title || 'Untitled List'
  const rawCategory = list.category || list.domain || 'General'
  const migratedCategory = migrateDomainName(rawCategory)
  const displayCategory = getDomainDisplayName(migratedCategory)
  const displayOwner = list.requester_name || list.owner_name || 'Unknown'
  const items = list.current_snapshot?.items || []

  // Get dynamic table columns from first item
  const tableColumns = items.length > 0 
    ? Object.keys(items[0]).filter(key => !['entry_id', 'version_id'].includes(key))
    : []
  
  // Format column header: contact_name -> Contact Name
  const formatColumnHeader = (key: string) => {
    return key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-slate-50">
      {/* Header */}
      <div className="sticky top-0 z-30 backdrop-blur-xl bg-white/80 border-b border-slate-200/60 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-5">
          <div className="flex items-center justify-between">
            <button 
              onClick={() => navigate('/dashboard')}
              className="flex items-center gap-2 text-slate-600 hover:text-slate-900 font-medium transition-colors group"
            >
              <svg className="w-5 h-5 group-hover:-translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7" />
              </svg>
              Back to Dashboard
            </button>
            <div className="flex gap-3">
              <button 
                onClick={() => setShowDeleteConfirm(true)}
                className="px-5 py-2.5 bg-red-500 hover:bg-red-600 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl hover:shadow-red-500/30 transition-all duration-300 hover:scale-105 flex items-center gap-2"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
                Delete List
              </button>
            </div>
          </div>
        </div>
      </div>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
        {/* Title Section */}
        <div className="mb-8">
          <div className="flex items-start justify-between mb-6">
            <div className="flex items-center gap-3 mb-3">
              <button 
                onClick={() => navigate('/dashboard')}
                className="flex items-center gap-2 text-slate-600 hover:text-slate-900 font-medium transition-colors group"
              >
                <svg className="w-5 h-5 group-hover:-translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7" />
                </svg>
                Back to Dashboard
              </button>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex-1">
                <div className="px-3 py-1.5 bg-gradient-to-r from-primary/10 to-secondary/10 text-primary text-sm tracking-wider uppercase font-semibold rounded-lg border border-primary/20 inline-block">
                  {displayCategory}
                </div>
              </div>
              <button
                onClick={handleRefresh}
                disabled={isRefreshing}
                className="p-2.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-600 hover:text-slate-900 transition-all duration-200 disabled:opacity-50"
                title="Refresh list"
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
          <div className="mb-6">
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center gap-1.5 text-slate-500 text-sm font-medium mb-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                  {displayOwner}
                </div>
                <h1 className="text-4xl font-bold text-slate-800 mb-2 bg-clip-text text-transparent bg-gradient-to-r from-slate-800 to-slate-600">
                  {displayTitle}
                </h1>
                <p className="text-slate-500 text-lg">
                  {items.length} items in this pharmaceutical list
                </p>
              </div>
            </div>
          </div>

          {/* Stats Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-sm font-medium text-slate-500 mb-1">Total Items</div>
                  <div className="text-3xl font-bold text-slate-800">{items.length}</div>
                </div>
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary/10 to-primary/5 flex items-center justify-center">
                  <svg className="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
              </div>
            </div>
            <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-sm font-medium text-slate-500 mb-1">Category</div>
                  <div className="text-xl font-bold text-slate-800">{displayCategory}</div>
                </div>
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-secondary/10 to-secondary/5 flex items-center justify-center">
                  <svg className="w-6 h-6 text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
                  </svg>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Data Table */}
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
          <div className="px-6 py-4 bg-gradient-to-r from-slate-50 to-white border-b border-slate-200">
            <h2 className="text-lg font-bold text-slate-800">List Items</h2>
          </div>
          
          {items.length === 0 ? (
            <div className="px-6 py-16 text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-slate-100 flex items-center justify-center">
                <svg className="w-8 h-8 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-slate-700 mb-2">No items yet</h3>
              <p className="text-slate-500 mb-4">Click the button below to add your first item</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-200">
                    {tableColumns.map(col => (
                      <th key={col} className="px-6 py-4 text-left text-xs font-bold text-slate-700 uppercase tracking-wider">
                        {formatColumnHeader(col)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {items.map((it: any, index: number) => (
                    <tr 
                      key={it.entry_id || index} 
                      className="hover:bg-gradient-to-r hover:from-slate-50 hover:to-transparent transition-colors duration-200 group"
                    >
                      {tableColumns.map(col => (
                        <td key={col} className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm text-slate-700">
                            {it[col] || 'N/A'}
                          </div>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          
          {/* Inline Add Entry Component - Note: This is a placeholder for list detail page */}
          {/* We need a table name that corresponds to the list's subdomain */}
          {list.subdomain && tableColumns.length > 0 && (
            <div className="mt-4 border-t border-slate-200 pt-4">
              <p className="text-sm text-slate-500 px-6 mb-4">
                Note: Adding entries here is currently not supported in list detail view. Please use the domain view to add entries.
              </p>
            </div>
          )}
        </div>
      </main>

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/40 backdrop-blur-sm"
            onClick={() => setShowDeleteConfirm(false)}
          />

          {/* Modal */}
          <div className="relative bg-white rounded-2xl shadow-2xl max-w-md w-full p-6 animate-fadeIn">
            {/* Icon */}
            <div className="mx-auto flex items-center justify-center w-14 h-14 rounded-full bg-red-100 mb-4">
              <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>

            {/* Content */}
            <div className="text-center mb-6">
              <h3 className="text-xl font-bold text-slate-800 mb-2">Delete List</h3>
              <p className="text-slate-600">
                Are you sure you want to delete <span className="font-semibold">"{displayTitle}"</span>? This action cannot be undone.
              </p>
            </div>

            {/* Actions */}
            <div className="flex gap-3">
              <button
                type="button"
                onClick={() => setShowDeleteConfirm(false)}
                className="flex-1 h-12 px-4 rounded-xl border-2 border-slate-200 bg-white text-slate-700 font-semibold hover:bg-slate-50 transition-all duration-200"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleDeleteList}
                className="flex-1 h-12 px-4 rounded-xl bg-red-500 hover:bg-red-600 text-white font-semibold shadow-lg hover:shadow-xl hover:shadow-red-500/30 transition-all duration-300 hover:scale-105"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Toast Notification */}
      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
    </div>
  )
}
