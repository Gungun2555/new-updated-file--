import axios from 'axios'

const baseURL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const instance = axios.create({ baseURL, timeout: 15000 })

instance.interceptors.request.use((config) => {
  const token = localStorage.getItem('sb_token')
  if (token && config.headers) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

export default instance
