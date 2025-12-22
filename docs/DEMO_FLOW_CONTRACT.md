# AURA Demo Flow Contract

## POST /ai/outfit-seed Pipeline

---

## 1. User Input

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `seed_image` | File | Photo of the seed garment |
| `gender` | string | `"male"` or `"female"` |

### Optional Fields

| Field | Type | Default | Options |
|-------|------|---------|---------|
| `event` | string | `null` | `work`, `date`, `party`, `casual` |
| `season` | string | `null` | `spring`, `summer`, `fall`, `winter` |
| `mode` | string | `"mock"` | `mock`, `partial_tryon`, `full_tryon` |
| `seed_category` | string | auto-detected | `top`, `bottom`, `outerwear`, `shoes` |
| `person_image` | File | `null` | Required for `full_tryon` mode |

---

## 2. Backend Response

```json
{
  "job_id": "uuid",
  "seed_locked": true,
  "subject_type": "mannequin | human",
  "tryon_mode": "mock | partial | full | partial_fallback | mixed",
  "seed": {
    "category": "top",
    "color": "black",
    "style": ["formal", "minimalist"],
    "image_path": "/path/to/seed.jpg",
    "detection_confidence": 0.92
  },
  "outfits": [
    {
      "rank": 1,
      "outfit_score": 0.847,
      "seed_locked": true,
      "items": {
        "top": { "id": "seed", "name": "User's Seed Item", "image_path": "..." },
        "bottom": { "id": "cat_101", "name": "Tailored Trousers", "color": "navy" },
        "outerwear": {},
        "shoes": {}
      },
      "lookbook_brand": "Zara",
      "matched_lookbook_rule": "power_dressing",
      "lookbook_bonus": 0.12,
      "lookbook_confidence": 0.9,
      "lookbook_bonus_applied": true,
      "trend_explanation": "This outfit follows Zara's Power Dressing approach, pairing structured outerwear with neutral tones while keeping your seed item fixed for coherence.",
      "render_url": "/ai/assets/jobs/{job_id}/renders/outfit_1.png",
      "tryon_mode": "partial"
    }
  ]
}
```

---

## 3. Frontend Rendering Requirements

### Seed Display

- Show seed image with **LOCK badge** overlay
- Display detected category and color
- Show detection confidence if < 0.85

### Outfit Cards (x5)

| Element | Source | Visual |
|---------|--------|--------|
| Rank | `outfit.rank` | #1, #2, etc. |
| Brand Badge | `outfit.lookbook_brand` | "ZARA" or "H&M" pill |
| Try-on Image | `outfit.render_url` | Main visual |
| Try-on Mode | `outfit.tryon_mode` | Label |
| Trend Explanation | `outfit.trend_explanation` | Italic text |
| Score | `outfit.outfit_score` | Progress bar |

### Brand Badge Colors

- **Zara**: Black background, white text
- **H&M**: Red background, white text

### Try-on Mode Labels

| Mode | Label | Color |
|------|-------|-------|
| `full` | "Full Try-On" | Green |
| `partial` | "Upper Body" | Blue |
| `partial_fallback` | "Fallback Render" | Orange |
| `mock` | "Preview" | Gray |

---

## 4. Demo Guarantees

1. **Seed Lock**: `seed_locked = true` always
2. **5 Outfits**: Always exactly 5 outfits returned
3. **Brand Mix**: MIN 2 Zara, MAX 2 H&M
4. **Render URL**: Always present (may be mock)
5. **Trend Explanation**: Present if LLM available, null otherwise

---

*Contract Version: 1.0.0*
