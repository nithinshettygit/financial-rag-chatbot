import { useCallback, useEffect, useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './App.css'

type ChatRole = 'user' | 'assistant'

interface ChatTurn {
  role: ChatRole
  content: string
}

interface SessionState {
  company: string | null
  fiscal_year: string | null
  target_metric: string | null
  document_id: string | null
}

interface ContextChunk {
  chunk_id: string
  content: string
  document_id?: string
  source_file?: string | null
  page_number?: number | string
  chunk_type?: string
  headings?: string[]
  final_score?: number
}

interface ChatResponse {
  session_id: string
  answer: string
  intent: string
  search_query: string
  state: SessionState
  retrieved_chunks: ContextChunk[]
  source_map: Record<string, string>
}

interface ChatMessage extends ChatTurn {
  retrievedChunks?: ContextChunk[]
  sourceMap?: Record<string, string>
}

interface RetryRequest {
  query: string
  history: ChatTurn[]
}

type ApiStatus = 'checking' | 'ready' | 'warming' | 'offline'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')
const SUGGESTIONS = [
  'What were Duolingo’s main sources of revenue?',
  'Summarize Nike’s key risk factors.',
  'Compare revenue across the reported fiscal years.',
]

const EMPTY_STATE: SessionState = {
  company: null,
  fiscal_year: null,
  target_metric: null,
  document_id: null,
}

function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`
}

function BotIcon() {
  return (
    <svg className="bot-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 4V2.5" />
      <circle cx="12" cy="2" r="1" />
      <rect x="4" y="7" width="16" height="14" rx="4" />
      <path d="M8.5 16h7" />
      <circle cx="9" cy="12" r="1" />
      <circle cx="15" cy="12" r="1" />
    </svg>
  )
}

function newSessionId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `session_${Date.now()}`
}

function isChatResponse(value: unknown): value is ChatResponse {
  if (typeof value !== 'object' || value === null) return false
  const response = value as Record<string, unknown>
  const state = response.state
  const chunks = response.retrieved_chunks
  const sourceMap = response.source_map

  return (
    typeof response.answer === 'string' &&
    typeof response.session_id === 'string' &&
    typeof state === 'object' &&
    state !== null &&
    'company' in state &&
    'fiscal_year' in state &&
    'target_metric' in state &&
    'document_id' in state &&
    Array.isArray(chunks) &&
    chunks.every(
      (chunk) =>
        typeof chunk === 'object' &&
        chunk !== null &&
        'chunk_id' in chunk &&
        typeof chunk.chunk_id === 'string' &&
        'content' in chunk &&
        typeof chunk.content === 'string',
    ) &&
    typeof sourceMap === 'object' &&
    sourceMap !== null &&
    Object.values(sourceMap).every((source) => typeof source === 'string')
  )
}

async function responseMessage(response: Response): Promise<string> {
  let payload: unknown
  try {
    payload = await response.json()
  } catch {
    return `The service returned an unexpected response (${response.status}).`
  }

  if (typeof payload === 'object' && payload !== null && 'detail' in payload) {
    const detail = (payload as { detail?: unknown }).detail
    if (typeof detail === 'string') {
      if (response.status === 503) return 'The service is warming up. Try again in a moment.'
      if (response.status === 504) return 'The request took too long. Please try again.'
      if (response.status === 429) return 'The service is busy. Please wait a moment and try again.'
      if (response.status === 401 || response.status === 403) {
        return 'This session is not authorized to use the service.'
      }
      return detail
    }
  }

  return `The request failed (${response.status}). Please try again.`
}

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [sessionId, setSessionId] = useState(newSessionId)
  const [sessionState, setSessionState] = useState<SessionState>(EMPTY_STATE)
  const [apiStatus, setApiStatus] = useState<ApiStatus>('checking')
  const [apiKey, setApiKey] = useState('')
  const [input, setInput] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [error, setError] = useState('')
  const [retryRequest, setRetryRequest] = useState<RetryRequest | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const conversationRef = useRef<HTMLDivElement>(null)
  const [isSidebarOpen, setIsSidebarOpen] = useState(
    () => window.matchMedia('(min-width: 681px)').matches,
  )

  useEffect(() => {
    const desktopQuery = window.matchMedia('(min-width: 681px)')
    const handleBreakpointChange = (event: MediaQueryListEvent) => {
      setIsSidebarOpen(event.matches)
    }

    desktopQuery.addEventListener('change', handleBreakpointChange)
    return () => desktopQuery.removeEventListener('change', handleBreakpointChange)
  }, [])

  useEffect(() => {
    if (!isSidebarOpen || window.matchMedia('(min-width: 681px)').matches) return

    const handleEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') setIsSidebarOpen(false)
    }

    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [isSidebarOpen])

  const checkReadiness = useCallback(async (signal?: AbortSignal) => {
    try {
      const response = await fetch(apiUrl('/health/ready'), {
        cache: 'no-store',
        signal,
      })
      setApiStatus(response.ok ? 'ready' : response.status === 503 ? 'warming' : 'offline')
    } catch (requestError) {
      if (requestError instanceof DOMException && requestError.name === 'AbortError') return
      setApiStatus('offline')
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    const initialCheckId = window.setTimeout(() => {
      void checkReadiness(controller.signal)
    }, 0)
    const intervalId = window.setInterval(() => void checkReadiness(), 30_000)

    return () => {
      controller.abort()
      window.clearTimeout(initialCheckId)
      window.clearInterval(intervalId)
    }
  }, [checkReadiness])

  useEffect(() => {
    if (messages.length === 0) return
    const conversation = conversationRef.current
    conversation?.scrollTo({ top: conversation.scrollHeight, behavior: 'smooth' })
  }, [messages, isSending])

  const startNewChat = () => {
    conversationRef.current?.scrollTo({ top: 0 })
    setMessages([])
    setSessionId(newSessionId())
    setSessionState(EMPTY_STATE)
    setInput('')
    setError('')
    setRetryRequest(null)
    if (window.matchMedia('(max-width: 680px)').matches) {
      setIsSidebarOpen(false)
    }
    textareaRef.current?.focus()
  }

  const sendMessage = async (query: string, history: ChatTurn[], addUserMessage: boolean) => {
    const normalizedQuery = query.trim()
    if (!normalizedQuery || !apiKey.trim() || normalizedQuery.length > 2000 || isSending) return

    if (addUserMessage) {
      setMessages((current) => [...current, { role: 'user', content: normalizedQuery }])
    }
    setIsSending(true)
    setError('')
    setRetryRequest(null)

    try {
      const response = await fetch(apiUrl('/v1/chat'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': apiKey.trim(),
        },
        body: JSON.stringify({
          session_id: sessionId,
          query: normalizedQuery,
          history,
          state: sessionState,
        }),
      })

      if (!response.ok) {
        throw new Error(await responseMessage(response))
      }

      const payload: unknown = await response.json()
      if (!isChatResponse(payload)) {
        throw new Error('The service returned an invalid chat response.')
      }

      setSessionState(payload.state)
      setMessages((current) => [
        ...current,
        {
          role: 'assistant',
          content: payload.answer,
          retrievedChunks: payload.retrieved_chunks,
          sourceMap: payload.source_map,
        },
      ])
    } catch (requestError) {
      const message =
        requestError instanceof Error
          ? requestError.message
          : 'Unable to reach the service. Check the connection and try again.'
      setError(message)
      setRetryRequest({ query: normalizedQuery, history })
    } finally {
      setIsSending(false)
      textareaRef.current?.focus()
    }
  }

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const query = input.trim()
    if (!query || !apiKey.trim() || apiStatus !== 'ready' || isSending) return

    const history = messages.map(({ role, content }) => ({ role, content }))
    setInput('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
    void sendMessage(query, history, true)
  }

  const handleInputKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      event.currentTarget.form?.requestSubmit()
    }
  }

  const handleInputChange = (value: string) => {
    setInput(value)
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`
    }
  }

  const statusLabel: Record<ApiStatus, string> = {
    checking: 'Connecting',
    ready: 'Ready',
    warming: 'Starting',
    offline: 'Offline',
  }

  return (
    <div className={`app-shell ${isSidebarOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
      <aside id="app-sidebar" className="sidebar" aria-label="Conversation controls">
        <div className="sidebar-header">
          <div className="brand">
            <div className="brand-mark"><BotIcon /></div>
            <span className="brand-name">AI Assistant</span>
          </div>
          <button
            className="sidebar-close"
            type="button"
            aria-label="Close sidebar"
            onClick={() => setIsSidebarOpen(false)}
          >
            <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
              <path d="m5 5 10 10M15 5 5 15" />
            </svg>
          </button>
        </div>

        <button className="new-chat-button" type="button" onClick={startNewChat}>
          <span className="plus-mark" aria-hidden="true">+</span>
          New chat
        </button>
        <div className="api-key-panel">
          <label htmlFor="api-key-input">API key</label>
          <input
            id="api-key-input"
            type="password"
            value={apiKey}
            onChange={(event) => setApiKey(event.target.value)}
            placeholder="Paste your client key"
            autoComplete="off"
            spellCheck={false}
          />
          <p>Your key stays in this tab and is sent only over HTTPS.</p>
        </div>
      </aside>

      <button
        className="sidebar-backdrop"
        type="button"
        aria-label="Close sidebar"
        onClick={() => setIsSidebarOpen(false)}
        tabIndex={isSidebarOpen ? 0 : -1}
      />

      <main className="main-panel">
        <header className="topbar">
          <div className="topbar-title">
            <button
              className="sidebar-toggle"
              type="button"
              aria-label={isSidebarOpen ? 'Close sidebar' : 'Open sidebar'}
              aria-controls="app-sidebar"
              aria-expanded={isSidebarOpen}
              onClick={() => setIsSidebarOpen((open) => !open)}
            >
              <span className="menu-icon" aria-hidden="true">
                <span className="menu-line menu-line-top" />
                <span className="menu-line menu-line-middle" />
                <span className="menu-line menu-line-bottom" />
              </span>
            </button>
            <strong className="page-title">Financial research</strong>
          </div>
          <div className={`connection-status connection-${apiStatus}`} role="status" aria-live="polite">
            <span className={`status-dot status-${apiStatus}`} aria-hidden="true" />
            {statusLabel[apiStatus]}
          </div>
        </header>

        <section className="chat-layout" aria-label="Financial research assistant">
          <div
            ref={conversationRef}
            className={`conversation ${messages.length === 0 ? 'conversation-empty' : ''}`}
          >
            {messages.length === 0 ? (
              <div className="welcome-panel">
                <h1>Research Duolingo & Nike Filings</h1>
                <div className="suggestion-area">
                  <p className="suggestion-label">Try asking</p>
                  <div className="suggestion-list">
                    {SUGGESTIONS.map((suggestion) => (
                      <button
                        className="suggestion-button"
                        key={suggestion}
                        type="button"
                        disabled={apiStatus !== 'ready' || isSending}
                        onClick={() => {
                          setInput(suggestion)
                          textareaRef.current?.focus()
                        }}
                      >
                        <span>{suggestion}</span>
                        <span className="suggestion-arrow" aria-hidden="true">↗</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="message-list">
                {messages.map((message, index) => (
                  <article className={`message message-${message.role}`} key={`${sessionId}-${index}`}>
                    {message.role === 'assistant' && (
                      <div className="assistant-mark"><BotIcon /></div>
                    )}
                    <div className="message-body">
                      {message.role === 'assistant' && (
                        <div className="assistant-name">AI Assistant</div>
                      )}
                      <div className="message-content">
                        {message.role === 'assistant' ? (
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                        ) : (
                          <p>{message.content}</p>
                        )}
                      </div>
                      {message.role === 'assistant' && message.retrievedChunks?.length ? (
                        <details className="sources">
                          <summary>
                            <span className="source-summary-label">Source passages</span>
                            <span className="source-count">{message.retrievedChunks.length}</span>
                            <span className="source-toggle" aria-hidden="true">+</span>
                          </summary>
                          <div className="source-list">
                            {message.retrievedChunks.map((chunk, sourceIndex) => {
                              const sourceId = `S${sourceIndex + 1}`
                              const sourceDescription = message.sourceMap?.[sourceId]
                              const section = chunk.headings?.filter(Boolean).join(' / ')
                              return (
                                <article className="source-item" key={chunk.chunk_id}>
                                  <div className="source-item-header">
                                    <span className="source-id">{sourceId}</span>
                                    <span className="source-file">{chunk.source_file ?? chunk.document_id ?? 'Filing'}</span>
                                    <span className="source-page">p. {chunk.page_number ?? '—'}</span>
                                  </div>
                                  {sourceDescription && <p className="source-description">{sourceDescription}</p>}
                                  {section && <p className="source-section">{section}</p>}
                                  <p className="source-excerpt">{chunk.content}</p>
                                </article>
                              )
                            })}
                          </div>
                        </details>
                      ) : null}
                    </div>
                  </article>
                ))}
                {isSending && (
                  <div className="loading-row" role="status" aria-live="polite">
                    <div className="assistant-mark"><BotIcon /></div>
                    <div className="loading-content">
                      <span className="thinking-label">Thinking…</span>
                      <span className="loading-indicator"><i /><i /><i /></span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="composer-wrap">
            {error && (
              <div className="error-banner" role="alert">
                <span>{error}</span>
                {retryRequest && (
                  <button
                    type="button"
                    onClick={() => void sendMessage(retryRequest.query, retryRequest.history, false)}
                    disabled={isSending || !apiKey.trim()}
                  >
                    Retry
                  </button>
                )}
              </div>
            )}
            <form className="composer" onSubmit={handleSubmit}>
              <label className="visually-hidden" htmlFor="question-input">Ask about a financial filing</label>
              <textarea
                id="question-input"
                ref={textareaRef}
                value={input}
                onChange={(event) => handleInputChange(event.target.value)}
                onKeyDown={handleInputKeyDown}
                placeholder="Ask about a company filing…"
                maxLength={2000}
                rows={1}
                disabled={apiStatus !== 'ready' || isSending}
              />
              <div className="composer-footer">
                <div className="composer-actions">
                  <span className="character-count">{input.length}/2000</span>
                  <button
                    className="send-button"
                    type="submit"
                    aria-label="Send question"
                    disabled={!input.trim() || !apiKey.trim() || apiStatus !== 'ready' || isSending}
                  >
                    <svg viewBox="0 0 20 20" aria-hidden="true">
                      <path d="M3.5 10h12M10 4.5l5.5 5.5-5.5 5.5" />
                    </svg>
                  </button>
                </div>
              </div>
            </form>
            <p className="disclaimer">
              {apiKey.trim()
                ? 'Check key figures against the source.'
                : 'Add your API key in the sidebar to start chatting.'}
            </p>
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
