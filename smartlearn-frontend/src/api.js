const API = import.meta.env.VITE_API_URL || "http://localhost:8000"

export const CHAT_ID = "day2-demo"

export async function uploadPDF(file) {
  const formData = new FormData()
  formData.append("file", file)

  const response = await fetch(`${API}/upload?chat_id=${encodeURIComponent(CHAT_ID)}`, {
    method: "POST",
    body: formData
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || "Upload failed")
  }

  return response.json()
}

export async function askQuestion(message, currentPage) {
  const body = { message, chat_id: CHAT_ID }
  if (currentPage) {
    body.current_page = currentPage
  }

  const response = await fetch(`${API}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || "Chat request failed")
  }

  return response.json()
}
