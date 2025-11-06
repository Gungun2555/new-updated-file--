import { useState } from 'react'
import { postListBotQuery, clearChatSession } from '../api/listbotApi'

export function useListBotChat() {
  const [messages, setMessages] = useState<{ role: 'user'|'assistant'; content: string }[]>([])
  const [loading, setLoading] = useState(false)
  const sessionId = 'default' // Could be made dynamic per domain

  const sendMessage = async (domain: string, query: string) => {
    setLoading(true)
    setMessages(prev => [...prev, { role: 'user', content: query }])
    try {
      // Convert messages to the format expected by the backend
      const chatHistory = messages.map(msg => ({
        role: msg.role,
        content: msg.content
      }))
      
      const res = await postListBotQuery({ 
        domain, 
        question: query, 
        chat_history: chatHistory,
        session_id: sessionId
      })
      
      setMessages(prev => [...prev, { role: 'assistant', content: res.answer }])
    } catch (err) {
      console.error('Chat error:', err)
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error processing your request. Please try again.'
      }])
    } finally {
      setLoading(false)
    }
  }

  const clearMessages = async () => {
    setMessages([])
    try {
      await clearChatSession(sessionId)
    } catch (err) {
      console.error('Error clearing session:', err)
    }
  }

  return { messages, sendMessage, clearMessages, loading }
}
