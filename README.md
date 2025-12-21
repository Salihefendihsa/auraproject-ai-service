# AuraProject AI Service v1.5.0

Production-ready AI outfit recommendation with observability.

## Features

- **Observability**: Request logging, metrics, cost tracking
- **Frontend**: Upload → Generate → View Try-On
- **MongoDB**: Persistent job storage
- **Caching**: Disk-based with 24h TTL
- **Hybrid LLM**: OpenAI + Gemini
- **Try-On**: SD Inpainting

## Quick Start

```powershell
# 1. Start MongoDB
docker run -d -p 27017:27017 --name mongo mongo:7

# 2. Set API key
$env:OPENAI_API_KEY = "sk-..."

# 3. Run
.\run.ps1

# 4. Open http://localhost:8000
```

## Observability & Metrics

### Request Logging

Requests are logged to `logs/requests.log`:
```json
{"timestamp":"2024-01-01T12:00:00Z","job_id":"abc","provider":"openai","cache_hit":false,"latency_ms":5000,"status":"success","tokens":1500,"cost_usd":0.003}
```

### Metrics Endpoint

`GET /metrics` returns:
```json
{
  "total_requests": 10,
  "cache_hits": 3,
  "cache_misses": 7,
  "cache_hit_ratio": 0.3,
  "total_tokens": 10500,
  "total_cost_usd": 0.021,
  "requests_by_provider": {"openai": 7, "cached": 3}
}
```

### Cost Tracking

Each job stores cost info:
```json
"cost": {
  "provider": "openai",
  "tokens": 1500,
  "estimated_usd": 0.003
}
```

### Health Endpoint

`GET /health` now includes:
```json
"observability": {
  "logging_enabled": true,
  "total_requests": 10,
  "cache_hit_ratio": 0.3,
  "total_cost_usd": 0.021
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | Required |
| `GEMINI_API_KEY` | - | Optional |
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB |
| `AURA_LOGGING_ENABLED` | `true` | Request logging |
| `AURA_CACHE_ENABLED` | `true` | Cache |

## Project Structure

```
ai_service/
├── frontend/
├── observability/    ← NEW
│   ├── logger.py
│   └── metrics.py
├── app/
├── cache/
├── config/
├── core/
├── db/
├── llm/
├── renderer/
└── vision/
logs/
└── requests.log
```
