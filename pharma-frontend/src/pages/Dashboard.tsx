import { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { getLists, createList } from '../api/listApi';
import NewListModal from '../components/NewListModal';
import Toast from '../components/Toast';
import { DOMAIN_CONFIGS, migrateDomainName, type DomainKey } from '../constants/domains';
import type { ListSummary } from '../types';

const Dashboard = () => {
  const navigate = useNavigate();
  const [lists, setLists] = useState<ListSummary[]>([]);
  const [isNewListModalOpen, setIsNewListModalOpen] = useState(false);
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' | 'warning' | 'info' } | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const fetchLists = useCallback(async () => {
    try {
      const res = await getLists();
      setLists(res);
    } catch (error) {
      console.error('Failed to fetch lists:', error);
    }
  }, []);

  useEffect(() => {
    void fetchLists();
  }, [fetchLists]);

  useEffect(() => {
    const pollInterval = setInterval(() => {
      void fetchLists();
    }, 30000);

    return () => clearInterval(pollInterval);
  }, [fetchLists]);

  useEffect(() => {
    const handleVisibilityChange = () => {
      if (!document.hidden) {
        void fetchLists();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
  }, [fetchLists]);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await fetchLists();
    setTimeout(() => setIsRefreshing(false), 500);
  };

  const getDomainStats = (domainKey: DomainKey) => {
    const domainConfig = DOMAIN_CONFIGS.find(d => d.key === domainKey);
    if (!domainConfig) return { count: 0, recentActivity: null };
    
    const domainLists = lists.filter((list: any) => {
      if (list.subdomain && list.subdomain.domain_id === domainConfig.domainId) {
        return true;
      }
      return false;
    });
    
    return {
      count: domainLists.length,
      recentActivity: domainLists.length > 0 ? domainLists[0].created_at : null
    };
  };

  const handleDomainClick = (domainKey: DomainKey) => {
    navigate(`/domain/${encodeURIComponent(domainKey)}`);
  };

  const handleCreateList = async (title: string, domain: string) => {
    try {
      const domainConfig = DOMAIN_CONFIGS.find(d => d.key === domain);
      if (!domainConfig) {
        setToast({ message: 'Invalid domain selected.', type: 'error' });
        return;
      }

      try {
        const { getSubdomains } = await import('../api/listApi');
        const subdomains = await getSubdomains(domainConfig.domainId);
        const matchingSubdomain = subdomains.find(s => s.subdomain_name === title);
        
        if (!matchingSubdomain) {
          setToast({ message: `Subdomain "${title}" not found in database. Please contact administrator.`, type: 'error' });
          return;
        }

        const newList = await createList({
          subdomain_id: matchingSubdomain.subdomain_id,
          requester_name: 'Current User',
          request_purpose: `${title} - ${new Date().toLocaleDateString()}`,
          status: 'In Progress'
        });

        if (newList?.request_id) {
          const updatedLists = await getLists();
          setLists(updatedLists);
          setIsNewListModalOpen(false);
          setToast({ message: 'List created successfully! You can now add entries from the domain view.', type: 'success' });
        } else {
          setToast({ message: 'Failed to create list. Please try again.', type: 'error' });
          setIsNewListModalOpen(false);
        }
      } catch (subdomainError) {
        console.error('Failed to fetch subdomains:', subdomainError);
        setToast({ message: 'Failed to fetch subdomain information. Please try again.', type: 'error' });
      }
    } catch (error) {
      console.error('Failed to create list:', error);
      setToast({ message: 'Failed to create list. Please try again.', type: 'error' });
      setIsNewListModalOpen(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-slate-50">
      {/* Modern Header with Glass Effect */}
      <header className="sticky top-0 z-40 backdrop-blur-xl bg-white/80 border-b border-slate-200/60 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-br from-primary to-secondary rounded-2xl blur-lg opacity-30"></div>
                <div className="relative flex items-center gap-3 px-4 py-2 bg-gradient-to-br from-primary to-secondary rounded-2xl shadow-lg">
                  <svg
                    className="w-7 h-7 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                  <h1 className="text-2xl font-bold text-white tracking-tight">
                    PharmaDB
                  </h1>
                </div>
              </div>
            </div>
            
            <div className="flex items-center gap-6">
              {/* IngestionHub */}
              {/* Ingestion Hub Button */}
  <button
    onClick={() => navigate('/ingestion-hub')}
    className="px-4 py-2 rounded-2xl bg-gradient-to-br from-primary to-secondary text-white font-semibold shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 flex items-center gap-2"
  >
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" />
    </svg>
    Ingestion Hub
  </button>

              <button
                onClick={handleRefresh}
                disabled={isRefreshing}
                className="p-2.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-600 hover:text-slate-900 transition-all duration-200 disabled:opacity-50"
                title="Refresh dashboard"
                type="button"
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
              

              <div className="h-8 w-px bg-slate-200 hidden md:block" />
              <button
                type="button"
                className="relative p-2.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-600 hover:text-slate-900 transition-all duration-200"
              >
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M5.121 17.804A13.937 13.937 0 0112 16c2.5 0 4.847.655 6.879 1.804M15 10a3 3 0 11-6 0 3 3 0 016 0zm6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
        {/* Hero Section */}
        <div className="mb-10">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
            <div>
              <h2 className="text-3xl font-bold text-slate-800 mb-2 bg-clip-text text-transparent bg-gradient-to-r from-slate-800 to-slate-600">
                Pharmaceutical Lists
              </h2>
              <p className="text-slate-500 text-lg">
                Select a domain to view and manage lists
              </p>
            </div>
          </div>

          {/* Stats Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
            <div className="relative overflow-hidden bg-white rounded-2xl p-5 border border-slate-200 hover:border-primary/30 hover:shadow-lg transition-all duration-300 group">
              <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-br from-primary/10 to-transparent rounded-bl-full"></div>
              <div className="relative">
                <div className="text-sm font-medium text-slate-500 mb-1">Total Lists</div>
                <div className="text-3xl font-bold text-slate-800">{lists.length}</div>
              </div>
            </div>
            <div className="relative overflow-hidden bg-white rounded-2xl p-5 border border-slate-200 hover:border-secondary/30 hover:shadow-lg transition-all duration-300 group">
              <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-br from-secondary/10 to-transparent rounded-bl-full"></div>
              <div className="relative">
                <div className="text-sm font-medium text-slate-500 mb-1">Active Domains</div>
                <div className="text-3xl font-bold text-slate-800">{DOMAIN_CONFIGS.length}</div>
              </div>
            </div>
            <div className="relative overflow-hidden bg-white rounded-2xl p-5 border border-slate-200 hover:border-success/30 hover:shadow-lg transition-all duration-300 group">
              <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-br from-success/10 to-transparent rounded-bl-full"></div>
              <div className="relative">
                <div className="text-sm font-medium text-slate-500 mb-1">Categories</div>
                <div className="text-3xl font-bold text-slate-800">{DOMAIN_CONFIGS.length}</div>
              </div>
            </div>
          </div>
        </div>

        {/* Domain Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {DOMAIN_CONFIGS.map((domain, index) => {
            const stats = getDomainStats(domain.key);
            const domainIcons = [
              // Customer/HCP icon
              <svg key="icon1" className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
              </svg>,
              // Account/Institutional icon
              <svg key="icon2" className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
              </svg>,
              // Marketing Campaign icon
              <svg key="icon3" className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 5.882V19.24a1.76 1.76 0 01-3.417.592l-2.147-6.15M18 13a3 3 0 100-6M5.436 13.683A4.001 4.001 0 017 6h1.832c4.1 0 7.625-1.234 9.168-3v14c-1.543-1.766-5.067-3-9.168-3H7a3.988 3.988 0 01-1.564-.317z" />
              </svg>,
              // Data/Analytics icon
              <svg key="icon4" className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            ];

            const gradients = [
              'from-blue-500 to-blue-600',
              'from-purple-500 to-purple-600',
              'from-pink-500 to-pink-600',
              'from-green-500 to-green-600'
            ];

            return (
              <div
                key={domain.key}
                onClick={() => handleDomainClick(domain.key)}
                className="group relative overflow-hidden bg-white rounded-2xl border border-slate-200 hover:border-primary/30 hover:shadow-2xl transition-all duration-500 hover:-translate-y-2 cursor-pointer animate-fadeIn"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                {/* Gradient accent on top */}
                <div className={`absolute top-0 left-0 right-0 h-2 bg-gradient-to-r ${gradients[index]} opacity-0 group-hover:opacity-100 transition-opacity duration-500`}></div>
                
                {/* Background gradient effect */}
                <div className={`absolute inset-0 bg-gradient-to-br ${gradients[index]} opacity-0 group-hover:opacity-5 transition-opacity duration-500`}></div>
                
                <div className="relative p-8">
                  {/* Icon and Title */}
                  <div className="flex items-start justify-between mb-6">
                    <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${gradients[index]} flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform duration-500`}>
                      {domainIcons[index]}
                    </div>
                    {/* View Lists Button - Moved to Top Right */}
                    <button
                      className="px-5 py-3 bg-gradient-to-r from-primary to-secondary text-white text-sm font-semibold rounded-xl shadow-lg hover:shadow-xl hover:shadow-primary/30 transition-all duration-300 hover:scale-105 flex items-center gap-2"
                    >
                      <span>View Lists</span>
                      <svg className="w-4 h-4 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                      </svg>
                    </button>
                  </div>

                  {/* Stats Badge */}
                  {stats.count > 0 && (
                    <div className="mb-4 inline-block px-3 py-1.5 bg-gradient-to-r from-primary/10 to-secondary/10 text-primary text-sm font-bold rounded-lg border border-primary/20">
                      {stats.count} {stats.count === 1 ? 'List' : 'Lists'}
                    </div>
                  )}

                  {/* Domain Name */}
                  <h3 className="text-2xl font-bold text-slate-800 mb-3 group-hover:text-primary transition-colors duration-300">
                    {domain.displayName}
                  </h3>

                  {/* List Types */}
                  <div className="mb-6 space-y-2">
                    {domain.listTypes.map((listType, idx) => (
                      <div key={idx} className="flex items-center gap-2 text-sm text-slate-600">
                        <div className="w-1.5 h-1.5 rounded-full bg-primary"></div>
                        <span>{listType}</span>
                      </div>
                    ))}
                  </div>

                  {/* Additional Action Buttons */}
                  <div className="flex gap-3">
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        navigate(`/domain/${encodeURIComponent(domain.key)}/requests`)
                      }}
                      className="flex-1 px-3 py-2 bg-white border-2 border-purple-500 text-purple-600 font-semibold rounded-xl hover:bg-purple-50 transition-all duration-300 flex items-center justify-center gap-2"
                      title="View all list requests for this domain"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                      </svg>
                      <span>Requests</span>
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        navigate(`/domain/${encodeURIComponent(domain.key)}/versions`)
                      }}
                      className="flex-1 px-3 py-2 bg-white border-2 border-orange-500 text-orange-600 font-semibold rounded-xl hover:bg-orange-50 transition-all duration-300 flex items-center justify-center gap-2"
                      title="View version history for this domain"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      <span>History</span>
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        navigate(`/domain/${encodeURIComponent(domain.key)}/history`)
                      }}
                      className="flex-1 px-3 py-2 bg-white border-2 border-primary text-primary font-semibold rounded-xl hover:bg-primary/5 transition-all duration-300 flex items-center justify-center gap-2"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      <span>Work Attribution</span>
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {lists.length === 0 && (
          <div className="text-center py-16 mt-8">
            <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-slate-100 flex items-center justify-center">
              <svg className="w-10 h-10 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-slate-800 mb-2">No lists yet</h3>
            <p className="text-slate-500">Create your first list to get started</p>
          </div>
        )}
      </main>

      {/* New List Modal */}
      <NewListModal 
        isOpen={isNewListModalOpen}
        onClose={() => setIsNewListModalOpen(false)}
        onSubmit={handleCreateList}
      />

      {/* Toast Notification */}
      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
    </div>
  );
};

export default Dashboard;