import { useEffect, useState } from 'react'
import { CHAT_ID } from './api.js'

export function PdfPreview({ currentPage, onPageChange }) {
  const [pdfUrl, setPdfUrl] = useState(null)

  useEffect(() => {
    const API = import.meta.env.VITE_API_URL || "http://localhost:8000"
    setPdfUrl(`${API}/documents/${encodeURIComponent(CHAT_ID)}/file`)
  }, [])

  const embedSrc = pdfUrl && currentPage ? `${pdfUrl}#page=${currentPage}` : pdfUrl

  return (
    <div className="preview-card">
      <h2>PDF Preview</h2>
      <div className="preview-content">
        {pdfUrl ? (
          <embed
            key={currentPage}
            className="pdf-embed"
            src={embedSrc}
            type="application/pdf"
            width="100%"
            height="100%"
          />
        ) : (
          <p>Upload a PDF to preview</p>
        )}
      </div>
    </div>
  )
}
