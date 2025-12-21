# AuraProject AI Service v1.4.3

MongoDB job persistence for production reliability.

## Features

- **MongoDB**: Persistent job storage
- **Caching**: Disk-based with 24h TTL
- **Hybrid LLM**: OpenAI + Gemini
- **Segmentation**: SegFormer
- **Attributes**: CLIP
- **Try-On**: SD Inpainting

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key |
| `GEMINI_API_KEY` | No | - | Gemini API key |
| `MONGO_URI` | No | `mongodb://localhost:27017` | MongoDB connection |
| `MONGO_DB_NAME` | No | `aura_ai` | Database name |
| `AURA_CACHE_ENABLED` | No | `true` | Enable caching |

## Quick Start

```powershell
# Start MongoDB (Docker)
docker run -d -p 27017:27017 --name mongo mongo:7

# Set API key
$env:OPENAI_API_KEY = "sk-..."

# Run
.\run.ps1
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service status (LLM, cache, MongoDB) |
| `/ai/outfit` | POST | Generate outfits (persisted to MongoDB) |
| `/ai/jobs/{id}` | GET | Retrieve persisted job |
| `/ai/assets/{path}` | GET | Serve images/renders |

## Health Check

```json
{
  "status": "ok",
  "version": "1.4.3",
  "mongo": {"status": "connected"},
  "cache": {"enabled": true},
  "llm": {"active_provider": "openai"}
}
```

## Job Persistence

Jobs survive server restarts:
```
POST /ai/outfit → job_id
[server restart]
GET /ai/jobs/{job_id} → full job data
```
