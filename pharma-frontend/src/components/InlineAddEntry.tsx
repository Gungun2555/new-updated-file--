import React, { useState, useEffect, useRef } from 'react'
import Toast from './Toast'
import { downloadSampleCSV, getListTypeFromSubdomain } from '../utils/csvTemplates'

interface InlineAddEntryProps {
  columns: string[]
  tableName: string
  onEntryAdded: () => void
  onAddingStateChange?: (isAdding: boolean) => void
  showAsButton?: boolean
  listId?: number // Optional list ID for CSV uploads
  subdomainName?: string // Optional subdomain name for CSV templates
}

export default function InlineAddEntry({ columns, tableName, onEntryAdded, onAddingStateChange, showAsButton = false, listId, subdomainName }: InlineAddEntryProps) {
  const [isAdding, setIsAdding] = useState(false)
  const [showMenu, setShowMenu] = useState(false)
  const [addMode, setAddMode] = useState<'manual' | 'csv' | null>(null)
  const [formData, setFormData] = useState<Record<string, string>>({})
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' | 'warning' | 'info' } | null>(null)
  const [isSaving, setIsSaving] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const menuRef = useRef<HTMLDivElement>(null)

  // Notify parent when adding state changes
  useEffect(() => {
    if (onAddingStateChange) {
      onAddingStateChange(isAdding)
    }
  }, [isAdding, onAddingStateChange])

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setShowMenu(false)
      }
    }

    if (showMenu) {
      document.addEventListener('mousedown', handleClickOutside)
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [showMenu])

  const handleInputChange = (column: string, value: string) => {
    setFormData(prev => ({ ...prev, [column]: value }))
  }

  const handlePlusClick = () => {
    setShowMenu(!showMenu)
  }

  const handleManualAdd = () => {
    setAddMode('manual')
    setIsAdding(true)
    setShowMenu(false)
    // Initialize form data with empty strings for all columns
    const initialData: Record<string, string> = {}
    columns.forEach(col => {
      initialData[col] = ''
    })
    setFormData(initialData)
  }

  const handleCsvAdd = () => {
    setAddMode('csv')
    setIsAdding(true)
    setShowMenu(false)
  }

  const handleCancel = () => {
    setIsAdding(false)
    setAddMode(null)
    setFormData({})
    setSelectedFile(null)
    setShowMenu(false)
  }

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      if (!file.name.endsWith('.csv')) {
        setToast({ message: 'Please select a CSV file', type: 'error' })
        return
      }
      setSelectedFile(file)
    }
  }

  const handleDownloadSample = () => {
    if (!subdomainName) {
      setToast({ message: 'Unable to determine list type for sample CSV', type: 'error' })
      return
    }
    
    const listType = getListTypeFromSubdomain(subdomainName)
    if (!listType) {
      setToast({ message: 'No sample template available for this list type', type: 'error' })
      return
    }
    
    try {
      downloadSampleCSV(listType)
      setToast({ message: 'Sample CSV downloaded successfully!', type: 'success' })
    } catch (error) {
      console.error('Error downloading sample CSV:', error)
      setToast({ message: 'Failed to download sample CSV', type: 'error' })
    }
  }

  const parseCsvFile = (file: File): Promise<Array<Record<string, string>>> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = (e) => {
        try {
          const text = e.target?.result as string
          const lines = text.split('\n').filter(line => line.trim())
          if (lines.length < 2) {
            reject(new Error('CSV file must have at least a header row and one data row'))
            return
          }
          
          const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''))
          const data = lines.slice(1).map(line => {
            const values = line.split(',').map(v => v.trim().replace(/^"|"$/g, ''))
            const row: Record<string, string> = {}
            headers.forEach((header, index) => {
              row[header] = values[index] || ''
            })
            return row
          })
          resolve(data)
        } catch (error) {
          reject(error)
        }
      }
      reader.onerror = () => reject(new Error('Failed to read file'))
      reader.readAsText(file)
    })
  }

  const handleCsvUpload = async () => {
    if (!selectedFile) {
      setToast({ message: 'Please select a CSV file', type: 'warning' })
      return
    }

    try {
      setIsUploading(true)

      // If listId is provided, use the list-based CSV upload endpoint
      if (listId) {
        const formData = new FormData()
        formData.append('file', selectedFile)

        console.log('[DEBUG] Uploading CSV to:', `http://localhost:8000/api/lists/${listId}/upload-csv`)

        const response = await fetch(`http://localhost:8000/api/lists/${listId}/upload-csv`, {
          method: 'POST',
          body: formData,
        })

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }))
          console.error('[ERROR] Server response:', errorData)
          throw new Error(errorData.detail || `Server error: ${response.status}`)
        }

        const result = await response.json()
        console.log('[SUCCESS] CSV uploaded:', result)

        setToast({ message: `Successfully uploaded ${result.items_added || 0} entries!`, type: 'success' })
      } else {
        // For standalone entries (no listId), parse CSV and add entries individually
        const rows = await parseCsvFile(selectedFile)
        console.log('[DEBUG] Parsed CSV rows:', rows.length)

        let successCount = 0
        let errorCount = 0

        for (const row of rows) {
          try {
            // Remove empty fields
            const dataToSend = Object.fromEntries(
              Object.entries(row).filter(([_, value]) => value.trim() !== '')
            )

            if (Object.keys(dataToSend).length === 0) {
              errorCount++
              continue
            }

            const response = await fetch(`http://localhost:8000/api/${tableName}`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify(dataToSend),
            })

            if (response.ok) {
              successCount++
            } else {
              errorCount++
              console.error(`[ERROR] Failed to add row:`, row)
            }
          } catch (error) {
            errorCount++
            console.error(`[ERROR] Error adding row:`, error)
          }
        }

        if (errorCount > 0) {
          setToast({ 
            message: `Uploaded ${successCount} entries successfully. ${errorCount} entries failed.`, 
            type: 'warning' 
          })
        } else {
          setToast({ message: `Successfully uploaded ${successCount} entries!`, type: 'success' })
        }
      }

      setIsAdding(false)
      setAddMode(null)
      setSelectedFile(null)
      onEntryAdded() // Refresh the table
    } catch (error) {
      console.error('[ERROR] Error uploading CSV:', error)
      const errorMessage = error instanceof Error ? error.message : 'Failed to upload CSV. Please try again.'
      setToast({ message: errorMessage, type: 'error' })
    } finally {
      setIsUploading(false)
    }
  }

  const handleSave = async () => {
    try {
      setIsSaving(true)
      
      // Remove empty fields
      const dataToSend = Object.fromEntries(
        Object.entries(formData).filter(([_, value]) => value.trim() !== '')
      )

      if (Object.keys(dataToSend).length === 0) {
        setToast({ message: 'Please fill in at least one field', type: 'warning' })
        setIsSaving(false)
        return
      }

      console.log('[DEBUG] Sending data to:', `http://localhost:8000/api/${tableName}`)
      console.log('[DEBUG] Data:', dataToSend)

      const response = await fetch(`http://localhost:8000/api/${tableName}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(dataToSend),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }))
        console.error('[ERROR] Server response:', errorData)
        throw new Error(errorData.detail || `Server error: ${response.status}`)
      }

      const result = await response.json()
      console.log('[SUCCESS] Entry added:', result)

      setToast({ message: 'Entry added successfully!', type: 'success' })
      setIsAdding(false)
      setFormData({})
      onEntryAdded() // Refresh the table
    } catch (error) {
      console.error('[ERROR] Error adding entry:', error)
      const errorMessage = error instanceof Error ? error.message : 'Failed to add entry. Please try again.'
      setToast({ message: errorMessage, type: 'error' })
    } finally {
      setIsSaving(false)
    }
  }

  // Format column header: contact_name -> Contact Name
  const formatColumnHeader = (key: string) => {
    return key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')
  }

  // If showing as button only (for top-right placement), return just the button with menu
  if (showAsButton && !isAdding) {
    return (
      <div className="relative" ref={menuRef}>
        <button
          onClick={handlePlusClick}
          className="px-5 py-2.5 bg-gradient-to-r from-primary to-secondary text-white font-semibold rounded-xl shadow-lg hover:shadow-xl hover:shadow-primary/30 transition-all duration-300 hover:scale-105 flex items-center gap-2"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" />
          </svg>
          Add
        </button>

        {/* Dropdown Menu */}
        {showMenu && (
          <div className="absolute right-0 mt-2 w-56 bg-white rounded-xl shadow-2xl border border-slate-200 z-50 overflow-hidden">
            <button
              onClick={handleManualAdd}
              className="w-full px-5 py-3 text-left hover:bg-gradient-to-r hover:from-primary/5 hover:to-secondary/5 transition-all flex items-center gap-3 border-b border-slate-100"
            >
              <svg className="w-5 h-5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              </svg>
              <div>
                <div className="font-semibold text-slate-800">Add Manually</div>
                <div className="text-xs text-slate-500">Enter data in form</div>
              </div>
            </button>
            <button
              onClick={handleCsvAdd}
              className="w-full px-5 py-3 text-left hover:bg-gradient-to-r hover:from-primary/5 hover:to-secondary/5 transition-all flex items-center gap-3"
            >
              <svg className="w-5 h-5 text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <div>
                <div className="font-semibold text-slate-800">Upload CSV</div>
                <div className="text-xs text-slate-500">Bulk import data</div>
              </div>
            </button>
          </div>
        )}
      </div>
    )
  }

  return (
    <>
      {!isAdding ? (
        showAsButton ? null : (
          <div className="p-6 flex justify-center">
            <div className="relative" ref={menuRef}>
              <button
                onClick={handlePlusClick}
                className="px-6 py-3 bg-gradient-to-r from-primary to-secondary text-white font-semibold rounded-xl shadow-lg hover:shadow-xl hover:shadow-primary/30 transition-all duration-300 hover:scale-105 flex items-center gap-2"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" />
                </svg>
                Add
              </button>

              {/* Dropdown Menu */}
              {showMenu && (
                <div className="absolute left-1/2 transform -translate-x-1/2 mt-2 w-56 bg-white rounded-xl shadow-2xl border border-slate-200 z-50 overflow-hidden">
                  <button
                    onClick={handleManualAdd}
                    className="w-full px-5 py-3 text-left hover:bg-gradient-to-r hover:from-primary/5 hover:to-secondary/5 transition-all flex items-center gap-3 border-b border-slate-100"
                  >
                    <svg className="w-5 h-5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                    </svg>
                    <div>
                      <div className="font-semibold text-slate-800">Add Manually</div>
                      <div className="text-xs text-slate-500">Enter data in form</div>
                    </div>
                  </button>
                  <button
                    onClick={handleCsvAdd}
                    className="w-full px-5 py-3 text-left hover:bg-gradient-to-r hover:from-primary/5 hover:to-secondary/5 transition-all flex items-center gap-3"
                  >
                    <svg className="w-5 h-5 text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    <div>
                      <div className="font-semibold text-slate-800">Upload CSV</div>
                      <div className="text-xs text-slate-500">Bulk import data</div>
                    </div>
                  </button>
                </div>
              )}
            </div>
          </div>
        )
      ) : addMode === 'manual' ? (
        <div className="border-t-2 border-primary/20 bg-gradient-to-r from-primary/5 to-secondary/5">
          <div className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-slate-800 flex items-center gap-2">
                <svg className="w-5 h-5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" />
                </svg>
                Adding New Entry
              </h3>
              <div className="flex gap-2">
                <button
                  onClick={handleCancel}
                  disabled={isSaving}
                  className="px-4 py-2 bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold rounded-lg transition-all disabled:opacity-50"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSave}
                  disabled={isSaving}
                  className="px-4 py-2 bg-gradient-to-r from-green-500 to-green-600 text-white font-semibold rounded-lg shadow-lg hover:shadow-xl hover:shadow-green-500/30 transition-all duration-300 hover:scale-105 disabled:opacity-50 flex items-center gap-2"
                >
                  {isSaving ? (
                    <>
                      <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Saving...
                    </>
                  ) : (
                    <>
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                      </svg>
                      Mark Done
                    </>
                  )}
                </button>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {columns.map((column) => (
                <div key={column}>
                  <label className="text-xs font-semibold text-slate-600 uppercase tracking-wider mb-2 block">
                    {formatColumnHeader(column)}
                  </label>
                  <input
                    type="text"
                    value={formData[column] || ''}
                    onChange={(e) => handleInputChange(column, e.target.value)}
                    placeholder={`Enter ${formatColumnHeader(column).toLowerCase()}`}
                    className="w-full px-3 py-2 rounded-lg border-2 border-slate-200 bg-white text-slate-700 focus:border-primary focus:ring-4 focus:ring-primary/10 transition-all duration-200"
                  />
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : addMode === 'csv' ? (
        <div className="border-t-2 border-secondary/20 bg-gradient-to-r from-secondary/5 to-primary/5">
          <div className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-slate-800 flex items-center gap-2">
                <svg className="w-5 h-5 text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                Upload CSV File
              </h3>
              <div className="flex gap-2">
                <button
                  onClick={handleCancel}
                  disabled={isUploading}
                  className="px-4 py-2 bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold rounded-lg transition-all disabled:opacity-50"
                >
                  Cancel
                </button>
                <button
                  onClick={handleCsvUpload}
                  disabled={isUploading || !selectedFile}
                  className="px-4 py-2 bg-gradient-to-r from-green-500 to-green-600 text-white font-semibold rounded-lg shadow-lg hover:shadow-xl hover:shadow-green-500/30 transition-all duration-300 hover:scale-105 disabled:opacity-50 flex items-center gap-2"
                >
                  {isUploading ? (
                    <>
                      <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Uploading...
                    </>
                  ) : (
                    <>
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                      </svg>
                      Upload
                    </>
                  )}
                </button>
              </div>
            </div>
            
            {/* File Upload Area */}
            <div className="mb-4">
              <label className="block mb-2">
                <div className="border-2 border-dashed border-slate-300 rounded-xl p-8 text-center hover:border-secondary hover:bg-secondary/5 transition-all cursor-pointer">
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileChange}
                    className="hidden"
                    id="csv-upload"
                  />
                  <label htmlFor="csv-upload" className="cursor-pointer">
                    <svg className="w-12 h-12 mx-auto mb-3 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    {selectedFile ? (
                      <div>
                        <p className="text-sm font-semibold text-slate-700 mb-1">Selected file:</p>
                        <p className="text-sm text-secondary font-medium">{selectedFile.name}</p>
                        <p className="text-xs text-slate-500 mt-2">Click to change file</p>
                      </div>
                    ) : (
                      <div>
                        <p className="text-sm font-semibold text-slate-700 mb-1">Click to select a CSV file</p>
                        <p className="text-xs text-slate-500">or drag and drop</p>
                      </div>
                    )}
                  </label>
                </div>
              </label>
            </div>

            {/* Download Sample CSV Button */}
            {subdomainName && (
              <div className="mb-4 flex justify-center">
                <button
                  onClick={handleDownloadSample}
                  className="px-4 py-2 bg-gradient-to-r from-blue-500 to-blue-600 text-white font-semibold rounded-lg shadow-md hover:shadow-lg hover:shadow-blue-500/30 transition-all duration-300 hover:scale-105 flex items-center gap-2"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10" />
                  </svg>
                  Download Sample CSV
                </button>
              </div>
            )}

            {/* Info Box */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <svg className="w-5 h-5 text-blue-500 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div className="text-sm text-blue-800">
                  <p className="font-semibold mb-1">CSV Upload Guidelines:</p>
                  <ul className="list-disc list-inside space-y-1 text-xs">
                    <li>Ensure your CSV file has the correct column headers</li>
                    <li>Check the sample CSV templates for reference</li>
                    <li>All entries will be validated before import</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {/* Toast Notification */}
      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
    </>
  )
}
