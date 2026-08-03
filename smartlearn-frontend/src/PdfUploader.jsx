import { useState } from 'react'

export function PdfUploader({ upload, status, onUpload }) {
  const [file, setFile] = useState(null)

  const handleUploadClick = () => {
    if (file) {
      onUpload(file)
    }
  }

  return (
    <div className="pdf-uploader">
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
        {status === "Uploading..." ? "Uploading..." : "Upload"}
      </button>
      {upload && (
        <div className="file-badge">
          📄 {upload.filename}
        </div>
      )}
    </div>
  )
}
