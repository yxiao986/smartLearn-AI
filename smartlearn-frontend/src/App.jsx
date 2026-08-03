import { useState } from 'react'
import { uploadPDF } from './api.js'
import { PdfUploader } from './PdfUploader.jsx'
import { ChatPanel } from './ChatPanel.jsx'
import { PdfPreview } from './PdfPreview.jsx'
import { useAsyncOperation } from './useAsyncOperation.js'

export default function App() {
  const [upload, setUpload] = useState(null)
  const [currentPage, setCurrentPage] = useState(null)
  const { execute, status, error } = useAsyncOperation()

  const handleUpload = (file) => {
    const onUploadSuccess = (result) => {
      setUpload(result)
      setCurrentPage(1)
    }
    execute("Uploading...", () => uploadPDF(file), onUploadSuccess)
  }

  const handleBusy = (isBusy) => {
    // ChatPanel signals when asking is in progress
  }

  const handleJumpToPage = (pageNumber) => {
    setCurrentPage(pageNumber)
  }

  return (
    <div className="app-root">
      <header className="app-header">
        <h1>Smartlearn</h1>
        <PdfUploader
          upload={upload}
          status={status}
          onUpload={handleUpload}
        />
        {error && <div className="header-alert">{error}</div>}
      </header>

      <main className="app-main">
        {upload ? (
          <div className="workspace">
            <div className="pdf-panel">
              <PdfPreview currentPage={currentPage} onPageChange={handleJumpToPage} />
            </div>
            <div className="chat-panel">
              <ChatPanel
                key={upload.filename}
                enabled={!!upload}
                onBusy={handleBusy}
                disabled={!upload}
                currentPage={currentPage}
                onJumpToPage={handleJumpToPage}
              />
            </div>
          </div>
        ) : (
          <div className="empty-state">
            <p>Upload a PDF to get started</p>
          </div>
        )}
      </main>
    </div>
  )
}
