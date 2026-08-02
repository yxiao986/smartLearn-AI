import { useState } from 'react'

export function useAsyncOperation() {
  const [status, setStatus] = useState(null)
  const [error, setError] = useState(null)

  const execute = async (statusMsg, asyncFn, onSuccess) => {
    setStatus(statusMsg)
    try {
      const result = await asyncFn()
      onSuccess(result)
      setError(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setStatus(null)
    }
  }

  return { execute, status, error }
}
