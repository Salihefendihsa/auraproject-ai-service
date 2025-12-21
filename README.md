# AuraProject AI Service v1.4.4

Full-stack AI outfit recommendation with virtual try-on.

## Features

- **Frontend Demo**: Upload → Generate → View Try-On
- **MongoDB**: Persistent job storage
- **Caching**: Disk-based with 24h TTL
- **Hybrid LLM**: OpenAI + Gemini
- **Segmentation**: SegFormer
- **Try-On**: SD Inpainting

## Quick Start

```powershell
# 1. Start MongoDB (Docker)
docker run -d -p 27017:27017 --name mongo mongo:7

# 2. Set API key
$env:OPENAI_API_KEY = "sk-..."

# 3. Run
.\run.ps1

# 4. Open browser
# http://localhost:8000
```

## Frontend Demo

Open `http://localhost:8000` in your browser:

1. **Upload** your photo
2. Click **Generate Outfits**
3. Wait for processing (polls every 2s)
4. View **5 outfit cards** with try-on renders

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend demo |
| `/health` | GET | Service status |
| `/ai/outfit` | POST | Generate outfits |
| `/ai/jobs/{id}` | GET | Get job by ID |
| `/ai/assets/{path}` | GET | Serve images |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | Required |
| `GEMINI_API_KEY` | - | Optional |
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB |

## Project Structure

```
ai_service/
├── frontend/         ← NEW
│   ├── index.html
│   ├── style.css
│   └── app.js
├── app/
├── cache/
├── config/
├── core/
├── db/
├── llm/
├── renderer/
└── vision/
```
