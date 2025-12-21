# AuraProject AI Service v1.1.1

Segmentation with attribute extraction + LLM-powered outfit recommendations.

## What's New in v1.1.1

- **Attribute Extraction**: Type, color, style for each detected item
- **CLIP Models**: OpenAI CLIP + Fashion-CLIP for classification
- **Enriched Response**: `detected_items` with full details

## Quick Start

```powershell
# 1. Setup (creates .venv)
.\setup.ps1

# 2. Set API key
$env:OPENAI_API_KEY = "sk-..."

# 3. Run
.\run.ps1
```

**Note:** First request downloads models (~1.5GB total).

## API Response

```json
{
  "job_id": "...",
  "detected_clothing": {"top": true, "bottom": true, ...},
  "detected_items": {
    "top": {
      "present": true,
      "type": "t-shirt",
      "color": "white",
      "style": "casual",
      "source": "user"
    },
    "outerwear": {
      "present": false
    }
  },
  "masks": {...},
  "outfits": [5 items],
  "note": "v1.1.1 - segmentation with attributes + LLM planning"
}
```

## Models Used

| Model | Purpose |
|-------|---------|
| mattmdjaga/segformer_b2_clothes | Clothing segmentation |
| openai/clip-vit-large-patch14 | Type + color classification |
| patrickjohncyh/fashion-clip | Style classification |

## Project Structure

```
ai_service/
├── app/
│   ├── main.py
│   └── routes.py
├── core/
│   ├── orchestrator.py
│   └── storage.py
└── vision/
    ├── segmenter.py    # SegFormer
    └── attributes.py   # CLIP extractors
```
