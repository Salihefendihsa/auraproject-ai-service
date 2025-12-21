# AuraProject AI Service v1.4.2

Request-level caching for cost reduction.

## Features

- **Caching**: Disk-based JSON cache with 24h TTL
- **Config**: Centralized settings from env vars
- **Segmentation**: SegFormer clothing detection
- **Attributes**: CLIP type/color/style
- **Hybrid LLM**: OpenAI + Gemini
- **Virtual Try-On**: SD Inpainting

## Caching

### How It Works

```
Request → Segmentation → Generate Cache Key → Check Cache
                                    ↓
                              Cache Hit? → Return Cached Response
                                    ↓ No
                              LLM + Rendering → Save to Cache
```

### Cache Key Generation

Key is SHA256 hash of:
- Input image hash
- Detected clothing (booleans)
- Detected attributes (type, color, style)
- User note
- Active LLM provider

### Cost Reduction

| Scenario | LLM Calls | Render Time |
|----------|-----------|-------------|
| First request | 1 | Full |
| Same image + context | 0 (cached) | 0 |

### TTL

Default: **24 hours** (1440 minutes)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key |
| `GEMINI_API_KEY` | - | Gemini API key |
| `AURA_LLM_ENABLED` | `true` | Enable LLM |
| `AURA_LLM_PRIMARY` | `openai` | Primary provider |
| `AURA_LLM_SECONDARY` | `gemini` | Secondary provider |
| `AURA_CACHE_ENABLED` | `true` | Enable caching |
| `AURA_CACHE_TTL_MINUTES` | `1440` | Cache TTL |

## Health Check

```json
{
  "status": "ok",
  "version": "1.4.2",
  "llm": {...},
  "cache": {
    "enabled": true,
    "type": "disk_json",
    "ttl_minutes": 1440,
    "entries": 5
  }
}
```

## API Response

```json
{
  "job_id": "...",
  "cache_hit": true,
  "outfits": [...],
  "note": "v1.4.2 - caching enabled for cost reduction"
}
```

## Project Structure

```
ai_service/
├── app/
├── cache/            ← NEW
│   ├── cache_store.py
│   └── cache_manager.py
├── config/
├── core/
├── llm/
├── renderer/
└── vision/
```
