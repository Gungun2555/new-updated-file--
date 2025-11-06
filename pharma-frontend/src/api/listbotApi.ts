import axiosClient from './axiosClient'

export interface ListBotQueryRequest {
  domain?: string
  question: string
  list_id?: string
  chat_history?: any[]
  session_id?: string
  request_id?: number
}

export interface ListBotQueryResponse {
  answer: string
  sources?: any[]
  generated_sql?: string
  row_count?: number
  query_type?: string
}

export const postListBotQuery = async (data: ListBotQueryRequest): Promise<ListBotQueryResponse> => {
  try {
    const response = await axiosClient.post('/api/chatbot/query', {
      question: data.question,
      chat_history: data.chat_history || [],
      session_id: data.session_id || 'default',
      request_id: data.request_id
    })
    return response.data
  } catch (error) {
    console.error('Error querying ListBot:', error)
    throw error
  }
}

export const clearChatSession = async (session_id: string = 'default'): Promise<void> => {
  try {
    await axiosClient.post('/api/chatbot/clear-session', null, {
      params: { session_id }
    })
  } catch (error) {
    console.error('Error clearing chat session:', error)
    throw error
  }
}

