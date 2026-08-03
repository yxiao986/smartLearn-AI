const API = import.meta.env.VITE_API_URL || "http://localhost:8000"

// One chat session per browser tab. The backend keeps documents in memory and
// drops them on restart, so a per-tab id is a closer match than a shared one:
// it also stops two tabs from overwriting each other's uploaded PDF.
function createChatId() {
  // randomUUID is only defined in a secure context (localhost counts, a plain
  // http LAN address does not).
  return crypto.randomUUID?.() ?? `chat-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
}

function getChatId() {
  let id = sessionStorage.getItem("smartlearn_chat_id")
  if (!id) {
    id = createChatId()
    sessionStorage.setItem("smartlearn_chat_id", id)
  }
  return id
}

export const CHAT_ID = getChatId()

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
