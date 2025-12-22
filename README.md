# AuraProject AI Service v2.7.0

Production-ready AI outfit recommendation with **Single Best Outfit Mode**.

## Version Highlights

| Version | Features |
|---------|----------|
| v2.0 | Security: validation, auth, rate limit |
| v2.1 | Context: event, weather, history |
| v2.2 | Try-On: mask quality, prompts, 2-stage |
| v2.3 | Wardrobe: upload, pHash, LLM context |
| v2.4 | Self-Critique: Gemini judge + OpenAI regen |
| v2.5 | LLM Config: GPT-5 & Gemini-3 ready |
| v2.6 | ControlNet: Pose locking for try-on |
| **v2.7** | **Single Best Outfit Mode** |

---

## Quick Start

### 1. Start Backend

```powershell
cd "c:\Users\SALİH\OneDrive\Desktop\AuraProject AI Service"
.\venv\Scripts\python.exe -m uvicorn ai_service.app.main:app --reload --port 8000
```

### 2. Create User & Get API Key

**PowerShell (curl.exe):**
```powershell
curl.exe -X POST "http://localhost:8000/ai/users" -F "name=demo"
```

**PowerShell (Invoke-RestMethod):**
```powershell
$body = @{ name = "demo" }
Invoke-RestMethod -Uri "http://localhost:8000/ai/users" -Method POST -Body $body
```

Response includes your `api_key` - save it for requests.

### 3. Check Health

```powershell
curl.exe http://localhost:8000/health
```

---

## Single Best Outfit Mode (v2.7)

Returns ONLY the highest-scored outfit with try-on render.

**PowerShell (curl.exe):**
```powershell
curl.exe -X POST "http://localhost:8000/ai/outfit" `
  -H "X-API-Key: YOUR_API_KEY" `
  -F "image=@photo.jpg" `
  -F "event=business" `
  -F "mode=single"
```

**PowerShell (Invoke-RestMethod):**
```powershell
$headers = @{ "X-API-Key" = "YOUR_API_KEY" }
$form = @{
    image = Get-Item -Path "photo.jpg"
    event = "business"
    mode = "single"
}
Invoke-RestMethod -Uri "http://localhost:8000/ai/outfit" -Method POST -Headers $headers -Form $form
```

---

## Full Mode (Default)

Returns 5 outfits with try-on renders.

```powershell
curl.exe -X POST "http://localhost:8000/ai/outfit" `
  -H "X-API-Key: YOUR_API_KEY" `
  -F "image=@photo.jpg" `
  -F "mode=full"
```

---

## Demo Frontend

Open `frontend/demo.html` in browser. Enter your API key when prompted.

---

## LLM Configuration

```powershell
# Switch planner to Gemini 3 Pro
$env:AURA_LLM_PROVIDER = "gemini"
$env:AURA_LLM_MODEL = "gemini-3-pro"

# Switch judge to Gemini 3 Pro
$env:AURA_JUDGE_PROVIDER = "gemini"
$env:AURA_JUDGE_MODEL = "gemini-3-pro"
```

---

## ControlNet Pose Lock

```powershell
$env:AURA_CONTROLNET_ENABLED = "true"
$env:AURA_CONTROLNET_TYPE = "pose"
```

Requires GPU. Falls back to SD Inpainting on CPU.

---

## Health Response (v2.7.0)

```json
{
  "version": "2.7.0",
  "product_mode_support": ["full", "single"],
  "controlnet": { "enabled": false, "type": "pose" }
}
```
