import { useState } from 'react'

export function ChatPanel({ upload, answer, status, onAsk }) {
  const [message, setMessage] = useState('')

  const handleAskClick = () => {
    const trimmed = message.trim()
    if (trimmed) {
      onAsk(trimmed)
    }
  }

  return (
    <section>
      <h2>Ask Question</h2>
      <label htmlFor="message-input">Question:</label>
      <textarea
        id="message-input"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Type your question..."
      />
      <button
        type="button"
        onClick={handleAskClick}
        disabled={!upload || !message.trim() || status}
      >
        Ask
      </button>
      {status === "Asking..." && <p>{status}</p>}
      {answer && (
        <div>
          <h3>Answer</h3>
          <p>{answer.answer}</p>
          {answer.citations && answer.citations.length > 0 && (
            <div>
              <strong>Citations:</strong>
              <div>
                {answer.citations.map((page) => (
                  <span key={page} style={{ marginRight: '0.5rem' }}>
                    Page {page}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  )
}
