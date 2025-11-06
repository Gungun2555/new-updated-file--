import { Routes, Route, Navigate } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import DomainView from './pages/DomainView'
import ListDetail from './pages/ListDetail'
import HistoryPage from './pages/HistoryPage'
import DomainHistoryPage from './pages/DomainHistoryPage'
import DomainVersionHistoryPage from './pages/DomainVersionHistoryPage'
import RequestContextPage from './pages/RequestContextPage'
import SidebarListBot from './components/SidebarListBot'

export default function App() {
  return (
    <div className="h-screen flex overflow-hidden bg-gradient-to-br from-slate-50 via-blue-50/20 to-slate-50">
      {/* AI Assistant Sidebar */}
      <aside className="flex-shrink-0 border-r border-slate-200 shadow-xl h-full">
        <SidebarListBot />
      </aside>
      
      {/* Main Content Area */}
      <main className="flex-1 min-w-0 overflow-auto relative">
        {/* Decorative background elements */}
        <div className="fixed inset-0 pointer-events-none overflow-hidden">
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-gradient-to-br from-primary/5 to-transparent rounded-full blur-3xl"></div>
          <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-gradient-to-tl from-secondary/5 to-transparent rounded-full blur-3xl"></div>
        </div>
        
        <div className="relative z-10">
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/domain/:domainKey" element={<DomainView />} />
            <Route path="/domain/:domainKey/history" element={<DomainHistoryPage />} />
            <Route path="/domain/:domainKey/versions" element={<DomainVersionHistoryPage />} />
            <Route path="/domain/:domainKey/requests" element={<RequestContextPage />} />
            <Route path="/list/:id" element={<ListDetail />} />
            <Route path="/list/:id/history" element={<HistoryPage />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}
