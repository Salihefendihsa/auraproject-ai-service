# AuraProject AI Service v1.2.0

Virtual try-on with Stable Diffusion Inpainting.

## Features

- **Segmentation**: SegFormer clothing detection
- **Attributes**: CLIP type/color/style extraction
- **LLM Planning**: OpenAI outfit recommendations
- **Virtual Try-On**: SD Inpainting for suggested items

## Quick Start

```powershell
# 1. Setup
.\setup.ps1

# 2. Set API key
$env:OPENAI_API_KEY = "sk-..."

# 3. Run
.\run.ps1
```

## First Run

Downloads models (~3GB total):
- SegFormer (~300MB)
- CLIP models (~1GB)  
- SD Inpainting (~2GB)

## Performance

| Device | Segmentation | Try-On (5 outfits) |
|--------|--------------|-------------------|
| GPU | ~3s | ~30s |
| CPU | ~15s | ~5-10min |

## API Response

```json
{
  "job_id": "...",
  "detected_items": {...},
  "outfits": [
    {
      "rank": 1,
      "items": {...},
      "render_url": "/ai/assets/jobs/{id}/renders/outfit_1.png",
      "tryon_method": "inpainting"
    }
  ],
  "note": "v1.2.0 - segmentation + attributes + LLM + try-on"
}
```

## Models Used

| Model | Purpose |
|-------|---------|
| mattmdjaga/segformer_b2_clothes | Segmentation |
| openai/clip-vit-large-patch14 | Type + color |
| patrickjohncyh/fashion-clip | Style |
| runwayml/stable-diffusion-inpainting | Try-on |

## Fallback Behavior

If try-on fails (GPU/model errors):
- Input image is copied as render output
- API still returns 200 OK
- `tryon_method` = "fallback"
