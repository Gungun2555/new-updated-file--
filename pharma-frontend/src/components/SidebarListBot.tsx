import React, { useState } from 'react'
import { MessageSquare, Send, ChevronLeft, ChevronRight, Sparkles } from 'lucide-react'
import { useListBotChat } from '../hooks/useListBotChat'

export default function SidebarListBot() {
  const [open, setOpen] = useState(true)
  const [input, setInput] = useState('')
  const { messages, sendMessage, clearMessages, loading } = useListBotChat()

  const handleSend = async () => {
    if (!input.trim()) return
    await sendMessage('', input)
    setInput('')
  }

  const sampleQuestions = [
    'Show all HCPs in the database',
    'Count all Target Lists entries',
    'Count all Call Lists entries',
  ]

  return (
    <div className={`h-full flex flex-col bg-gradient-to-br from-slate-50 to-white border-l border-slate-200 shadow-xl ${open ? 'w-[420px]' : 'w-[70px]'} transition-all duration-300`}>
      <div className="relative overflow-hidden border-b border-slate-200 flex-shrink-0">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-secondary/5"></div>
        <div className={`relative p-5 flex items-center ${open ? 'justify-between' : 'justify-center flex-col gap-3'}`}>
          <div className={`relative ${!open ? 'cursor-pointer' : ''}`} onClick={() => !open && setOpen(true)}>
            <div className="absolute inset-0 bg-gradient-to-br from-primary to-secondary rounded-xl blur-md opacity-40"></div>
            <div className="relative w-11 h-11 rounded-xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center shadow-lg">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
          </div>
          {open && (
            <div className="flex items-center gap-4 flex-1 ml-2">
              <div>
                <h3 className="text-base font-bold text-slate-800 flex items-center gap-2">
                  AI Assist
                </h3>
                <p className="text-xs text-slate-500 mt-0.5">Ask questions about your data</p>
              </div>
            </div>
          )}
          <button
            onClick={() => setOpen(v => !v)}
            className="p-2 rounded-lg hover:bg-slate-100 text-slate-500 hover:text-slate-700 transition-all duration-200 flex-shrink-0"
            title={open ? 'Collapse' : 'Expand'}
          >
            {open ? <ChevronRight className="w-5 h-5" /> : <ChevronLeft className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {open && (
        <>
          <div className="flex-1 overflow-y-auto p-5 space-y-4">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center px-4">
                <div className="relative mb-4">
                  <div className="absolute inset-0 bg-gradient-to-br from-primary to-secondary rounded-2xl blur-xl opacity-20"></div>
                  <div className="relative w-16 h-16 rounded-2xl bg-gradient-to-br from-primary/10 to-secondary/10 flex items-center justify-center border border-primary/20">
                    <MessageSquare className="w-8 h-8 text-primary" />
                  </div>
                </div>
                <h4 className="text-base font-bold text-slate-800 mb-2">AI Database Assistant</h4>
                <p className="text-sm text-slate-500 max-w-xs leading-relaxed mb-6">
                  Ask natural language questions about your pharmaceutical lists, HCPs, versions, and more.
                </p>
                
                <div className="space-y-2 w-full">
                  <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Try asking:</p>
                  {sampleQuestions.map((prompt) => (
                    <button
                      key={prompt}
                      onClick={() => setInput(prompt)}
                      className="w-full p-3 text-left text-sm text-slate-600 bg-white hover:bg-gradient-to-r hover:from-primary/5 hover:to-secondary/5 rounded-lg border border-slate-200 hover:border-primary/30 transition-all duration-200"
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <>
                {messages.map((m, i) => (
                  <div key={i} className={`flex items-start gap-3 ${m.role === 'assistant' ? 'flex-row' : 'flex-row-reverse'}`}>
                    <div className={`w-9 h-9 rounded-xl flex items-center justify-center shrink-0 shadow-sm
                                ${m.role === 'assistant' 
                                  ? 'bg-gradient-to-br from-primary/10 to-secondary/10 border border-primary/20' 
                                  : 'bg-gradient-to-br from-slate-100 to-slate-50 border border-slate-200'}`}>
                      {m.role === 'assistant' ? (
                        <Sparkles className="w-4 h-4 text-primary" />
                      ) : (
                        <svg className="w-4 h-4 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" 
                                d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                        </svg>
                      )}
                    </div>
                    <div className={`flex-1 max-w-[85%] ${m.role === 'assistant' ? '' : 'text-right'}`}>
                      <div className={`inline-block p-4 rounded-2xl shadow-sm ${
                        m.role === 'assistant' 
                          ? 'bg-white border border-slate-200 rounded-tl-sm' 
                          : 'bg-gradient-to-br from-primary to-secondary text-white rounded-tr-sm'
                      }`}>
                        <div className={`text-sm leading-relaxed whitespace-pre-wrap ${m.role === 'assistant' ? 'text-slate-700' : 'text-white'}`}>
                          {m.content}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
                {loading && (
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary/10 to-secondary/10 border border-primary/20 flex items-center justify-center">
                      <Sparkles className="w-4 h-4 text-primary animate-pulse" />
                    </div>
                    <div className="flex items-center gap-2 p-4 bg-white border border-slate-200 rounded-2xl rounded-tl-sm">
                      <div className="flex gap-1">
                        <div className="w-2 h-2 rounded-full bg-primary/60 animate-bounce" style={{ animationDelay: '0ms' }}></div>
                        <div className="w-2 h-2 rounded-full bg-primary/60 animate-bounce" style={{ animationDelay: '150ms' }}></div>
                        <div className="w-2 h-2 rounded-full bg-primary/60 animate-bounce" style={{ animationDelay: '300ms' }}></div>
                      </div>
                      <span className="text-sm text-slate-500 ml-2">Thinking...</span>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          <div className="p-5 border-t border-slate-200 bg-gradient-to-br from-slate-50/50 to-transparent flex-shrink-0">
            <div className="flex gap-2 bg-white rounded-2xl border-2 border-slate-200 focus-within:border-primary focus-within:ring-4 focus-within:ring-primary/10 transition-all duration-200 p-1.5 shadow-sm">
              <input 
                type="text" 
                value={input} 
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSend()}
                className="flex-1 px-3 py-2.5 bg-transparent border-none focus:outline-none text-slate-700 placeholder:text-slate-400 font-medium" 
                placeholder="Ask anything about your data..." 
              />
              <button 
                onClick={handleSend} 
                disabled={!input.trim()}
                className="p-3 rounded-xl bg-gradient-to-br from-primary to-secondary text-white disabled:opacity-40 disabled:cursor-not-allowed
                         hover:shadow-lg hover:shadow-primary/30 transition-all duration-200 disabled:hover:shadow-none flex-shrink-0"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
