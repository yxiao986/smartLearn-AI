# Day 2 Test Log

**Date**: 2026-08-02  
**Status**: ✅ Checkpoint 5 Passed

---

## Test Results

| Test Case | Expected | Actual | Status |
| :--- | :--- | :--- | :---: |
| **PDF Upload** | Returns 200 & page count | Text extracted successfully | ✅ PASS |
| **Chat & Citation** | Answers with `[Page X]` | Correct page citations rendered | ✅ PASS |
| **Backend Down** | UI shows clear network error | Readable error message shown | ✅ PASS |
| **Backend Restart** | Prompts to re-upload PDF | Intercepted & re-upload requested | ✅ PASS |

---

## Verification
- **Frontend**: `http://localhost:5173`
- **Backend**: `http://127.0.0.1:8000`
- **Result**: E2E pipeline & error recovery verified.