# AuraProject AI Service v1.1.0

Real clothing segmentation + LLM-powered outfit recommendations.

## Quick Start

```powershell
# 1. Setup (creates .venv)
.\setup.ps1

# 2. Set API key
$env:OPENAI_API_KEY = "sk-..."

# 3. Run
.\run.ps1
```

**Note:** First run downloads SegFormer model (~300MB).

## API Endpoints

### GET /health
```json
{"status": "ok", "llm_configured": true, "provider": "openai"}
```

### POST /ai/outfit
Upload an image, get 5 outfit recommendations.

**Request:** `multipart/form-data`
- `image` (file, required)
- `user_note` (string, optional)

**Response:**
```json
{
  "job_id": "uuid",
  "seed": {"input_image": "/ai/assets/jobs/{id}/input.jpg"},
  "detected_clothing": {"top": true, "bottom": false, ...},
  "masks": {"top": "/ai/assets/jobs/{id}/masks/mask_top.png"},
  "raw_labels": ["Upper-clothes", "Pants", ...],
  "outfits": [5 outfit objects],
  "note": "v1.1.0 - segmentation + LLM planning active"
}
```

### GET /ai/assets/{path}
Serve saved images and masks.

## Project Structure

```
AuraProject AI Service/
├── ai_service/
│   ├── app/
│   │   ├── main.py
│   │   └── routes.py
│   ├── core/
│   │   ├── orchestrator.py
│   │   └── storage.py
│   └── vision/
│       └── segmenter.py
├── setup.ps1
├── run.ps1
└── requirements.txt
```
