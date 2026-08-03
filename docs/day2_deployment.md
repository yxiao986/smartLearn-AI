# Day 2 Deployment

## URLs
- Frontend: https://smartlearn-aibyxyf.vercel.app/
- Backend health: https://smartlearn-ai-production-87bf.up.railway.app/health
- Backend docs: https://smartlearn-ai-production-87bf.up.railway.app/docs

## Source
- Repository: student's fork
- Deployed branch / merge target: main
- Merged commit: ...
- Pull Request: ...

## Root Directories
- Railway: smartlearn-backend
- Vercel: smartlearn-frontend

## Environment variable names
- Railway: OPENROUTER_API_KEY, ALLOWED_ORIGINS
- Vercel: VITE_API_URL

## Acceptance results
- /health: pass/fail
- Upload: pass/fail
- Known /chat + citations: pass/fail + expected page
- Unknown question: pass/fail
- CORS restart + re-upload recovery: pass/fail

## Known limitations
- Railway restart clears in-memory uploaded/chat state; re-upload is expected.