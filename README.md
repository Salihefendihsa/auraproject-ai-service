# AuraProject AI Service v1.3.0

Hybrid LLM brain with OpenAI + Gemini.

## Features

- **Segmentation**: SegFormer clothing detection
- **Attributes**: CLIP type/color/style extraction
- **Hybrid LLM**: OpenAI (planner) + Gemini (advisor)
- **Virtual Try-On**: SD Inpainting rendering

## Hybrid LLM Logic

```
┌─────────────────────────────────────────┐
│           HYBRID LLM BRAIN              │
├─────────────────────────────────────────┤
│  1. Gemini (optional, advisory)         │
│     → Analyzes trends & context         │
│     → Returns 1-2 sentence advice       │
│                                         │
│  2. OpenAI (required, primary)          │
│     → Receives Gemini context           │
│     → Generates 5 complete outfits      │
└─────────────────────────────────────────┘
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | **Yes** | Primary planner |
| `GEMINI_API_KEY` | No | Style advisor |

## Quick Start

```powershell
# Required
$env:OPENAI_API_KEY = "sk-..."

# Optional (enhances recommendations)
$env:GEMINI_API_KEY = "AIza..."

# Run
.\run.ps1
```

## Fallback Behavior

| Scenario | Behavior |
|----------|----------|
| Gemini fails | Continue with OpenAI only |
| Gemini not configured | Use OpenAI only |
| OpenAI fails | Return error (primary required) |

## Health Check Response

```json
{
  "status": "ok",
  "version": "1.3.0",
  "llm": {
    "openai": true,
    "gemini": true
  }
}
```

## Project Structure

```
ai_service/
├── app/
├── core/
├── llm/              ← NEW
│   ├── openai_client.py
│   ├── gemini_client.py
│   └── router.py
├── renderer/
└── vision/
```
