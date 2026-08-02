import { useState } from 'react'
import { uploadPDF, askQuestion } from './api.js'
import { PdfUploader } from './PdfUploader.jsx'
import { ChatPanel } from './ChatPanel.jsx'
import { useAsyncOperation } from './useAsyncOperation.js'

export default function App() {
  const [upload, setUpload] = useState(null)
  const [answer, setAnswer] = useState(null)
  const { execute, status, error } = useAsyncOperation()

  const handleUpload = (file) => {
    execute("Uploading...", () => uploadPDF(file), setUpload)
  }

  const handleAsk = (message) => {
    execute("Asking...", () => askQuestion(message), setAnswer)
  }

  return (
    <div className="container">
      <h1>Smartlearn</h1>
      <form onSubmit={(e) => e.preventDefault()}>
        <PdfUploader
          upload={upload}
          status={status}
          onUpload={handleUpload}
        />

        <ChatPanel
          upload={upload}
          answer={answer}
          status={status}
          onAsk={handleAsk}
        />

        {error && <div role="alert" style={{ color: 'red' }}>{error}</div>}
      </form>
    </div>
  )
}
