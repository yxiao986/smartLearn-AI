import { useState } from 'react'

export function PdfUploader({ upload, status, onUpload }) {
  const [file, setFile] = useState(null)

  const handleUploadClick = () => {
    if (file) {
      onUpload(file)
    }
  }

  return (
    <section>
      <h2>Upload PDF</h2>
      <label htmlFor="pdf-input">Select PDF:</label>
      <input
        id="pdf-input"
        type="file"
        accept=".pdf"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
      />
      <button
        type="button"
        onClick={handleUploadClick}
        disabled={!file || status}
      >
        Upload
      </button>
      {status === "Uploading..." && <p>{status}</p>}
      {upload && (
        <div>
          <p>✓ Uploaded: {upload.filename}</p>
          <p>Pages: {upload.pages} | Characters: {upload.characters}</p>
        </div>
      )}
    </section>
  )
}
