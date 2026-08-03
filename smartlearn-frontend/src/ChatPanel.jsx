import { useState, useEffect, useRef } from 'react'
import { askQuestion } from './api.js'

const SUGGESTIONS = [
  'Summarize this paper',
  'What are the main findings?',
  'Tell me more about this page',
]

export function ChatPanel({ enabled, onBusy, disabled, currentPage, onJumpToPage }) {
  const [message, setMessage] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    setMessages([])
  }, [enabled])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleAskClick = async () => {
    const trimmed = message.trim()
    if (trimmed) {
      setMessage('')
      setLoading(true)
      setError(null)
      if (onBusy) onBusy(true)
      try {
        const answer = await askQuestion(trimmed, currentPage)
        setMessages([...messages, { question: trimmed, ...answer }])
      } catch (err) {
        setError(err.message)
        console.error(err)
      } finally {
        setLoading(false)
        if (onBusy) onBusy(false)
      }
    }
  }

  const handleSuggestionClick = (suggestion) => {
    setMessage(suggestion)
  }

  const handleCitationClick = (page) => {
    if (onJumpToPage) {
      onJumpToPage(page)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleAskClick()
    }
  }

  return (
    <section className="chat-section">
      <h2>Chat</h2>
      <div className="messages">
        {messages.length === 0 ? (
          <div className="chat-empty-state">
            <div className="empty-icon">💬</div>
            <h3>Ask questions about your PDF</h3>
            <div className="suggestion-chips">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  className="suggestion-chip"
                  onClick={() => handleSuggestionClick(suggestion)}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg, idx) => (
              <div key={idx} className="message">
                <div className="question">
                  <strong>Q:</strong> {msg.question}
                </div>
                <div className="answer">
                  <strong>A:</strong> {msg.answer}
                </div>
                {msg.citations && msg.citations.length > 0 && (
                  <div className="citations">
                    {msg.citations.map((page) => (
                      <button
                        key={page}
                        className="citation-button"
                        onClick={() => handleCitationClick(page)}
                      >
                        Page {page}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      <div className="input-error">
        {error && <p>{error}</p>}
      </div>

      <div className="input-area">
        <textarea
          id="message-input"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask a question..."
          disabled={disabled || loading}
        />
        <button
          className="send-button"
          onClick={handleAskClick}
          disabled={disabled || !message.trim() || loading}
          title="Send message (Shift+Enter for new line)"
        >
          {loading ? '⏳' : '↑'}
        </button>
      </div>
    </section>
  )
}
