# AuraProject AI Service v1.4.1

Centralized API key and LLM provider management.

## Features

- **Config System**: Centralized settings from env vars
- **Provider Management**: Priority, fallback, availability
- **Segmentation**: SegFormer clothing detection
- **Attributes**: CLIP type/color/style
- **Hybrid LLM**: OpenAI + Gemini
- **Virtual Try-On**: SD Inpainting

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key (primary planner) |
| `GEMINI_API_KEY` | - | Gemini API key (advisor) |
| `AURA_LLM_ENABLED` | `true` | Enable/disable LLM |
| `AURA_LLM_PRIMARY` | `openai` | Primary provider |
| `AURA_LLM_SECONDARY` | `gemini` | Secondary/fallback |
| `AURA_LLM_DAILY_LIMIT` | `200` | Daily request limit |

## Quick Start

```powershell
# Required
$env:OPENAI_API_KEY = "sk-..."

# Optional
$env:GEMINI_API_KEY = "AIza..."
$env:AURA_LLM_PRIMARY = "openai"
$env:AURA_LLM_SECONDARY = "gemini"

# Run
.\run.ps1
```

## Fallback Behavior

| Scenario | Behavior |
|----------|----------|
| Primary fails | Try secondary provider |
| Secondary fails | Return error |
| LLM disabled | Return error immediately |
| No API keys | Return error |

## Health Check Response

```json
{
  "status": "ok",
  "version": "1.4.1",
  "llm": {
    "enabled": true,
    "primary": "openai",
    "secondary": "gemini",
    "availability": {"openai": true, "gemini": false},
    "active_provider": "openai",
    "daily_limit": 200
  }
}
```

## Project Structure

```
ai_service/
├── app/
├── config/           ← NEW
│   ├── settings.py
│   └── providers.py
├── core/
├── llm/
├── renderer/
└── vision/
```
