"""System Check-Up Script"""
import sys
sys.path.insert(0, 'ai_service')

print('='*60)
print('AURA SYSTEM CHECK-UP')
print('='*60)

# 1. CUDA / GPU Check
print('\n[1] GPU STATUS')
try:
    import torch
    cuda = torch.cuda.is_available()
    print(f'  CUDA Available: {cuda}')
    if cuda:
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'  VRAM: {mem:.1f} GB')
    else:
        print('  WARNING: NO CUDA - GPU rendering will NOT work!')
except Exception as e:
    print(f'  ERROR PyTorch: {e}')

# 2. MongoDB Check
print('\n[2] DATABASE STATUS')
try:
    from db import mongo
    status = mongo.health_check()
    connected = status.get('connected', False)
    database = status.get('database', 'unknown')
    print(f'  MongoDB Connected: {connected}')
    print(f'  Database: {database}')
except Exception as e:
    print(f'  ERROR MongoDB: {e}')

# 3. LLM API Keys Check
print('\n[3] LLM API KEYS')
import os
openai_key = os.getenv('OPENAI_API_KEY', '')
gemini_key = os.getenv('GEMINI_API_KEY', '')
has_openai = 'SET' if openai_key else 'MISSING'
has_gemini = 'SET' if gemini_key else 'MISSING'
print(f'  OpenAI Key: {has_openai}')
print(f'  Gemini Key: {has_gemini}')

# 4. Core Models Check
print('\n[4] AI MODELS')
try:
    from vision.segmenter import segmenter
    print('  SegFormer: Loaded OK')
except Exception as e:
    print(f'  SegFormer ERROR: {e}')

try:
    from renderer.tryon import TryOnRenderer
    renderer = TryOnRenderer()
    print('  SD Inpainting: Ready (lazy load)')
except Exception as e:
    print(f'  SD Inpainting ERROR: {e}')

# 5. Catalog Check
print('\n[5] CATALOG DATA')
try:
    from outfit_recommender import load_catalog
    catalog = load_catalog()
    items = catalog.get('items', [])
    print(f'  Catalog Items: {len(items)}')
    if items:
        categories = set(item.get('category') for item in items)
        print(f'  Categories: {categories}')
except Exception as e:
    print(f'  Catalog ERROR: {e}')

# 6. Model Images Check
print('\n[6] MODEL IMAGES')
from pathlib import Path
male = Path('ai_service/data/models/model_male.jpg')
female = Path('ai_service/data/models/model_female.jpg')
male_ok = 'OK' if male.exists() else 'MISSING'
female_ok = 'OK' if female.exists() else 'MISSING'
print(f'  Male Model: {male_ok}')
print(f'  Female Model: {female_ok}')

# 7. Garment Images Check
print('\n[7] GARMENT IMAGES')
garment_dir = Path('ai_service/static/garments')
if garment_dir.exists():
    garments = list(garment_dir.glob('*.jpg'))
    print(f'  Garment Images: {len(garments)}')
else:
    print('  Garment Dir: MISSING')

# 8. Server Check
print('\n[8] SERVER STATUS')
try:
    import requests
    resp = requests.get('http://localhost:8000/health', timeout=5)
    if resp.status_code == 200:
        print('  Server: Running OK')
        health = resp.json()
        llm_info = health.get('llm', {})
        active = llm_info.get('active_provider', 'unknown')
        print(f'  LLM Active: {active}')
    else:
        print(f'  Server: Status {resp.status_code}')
except Exception as e:
    print(f'  Server ERROR: {e}')

# 9. End-to-End Test
print('\n[9] END-TO-END TEST')
try:
    from outfit_recommender import generate_outfit_combos
    fixed = {'name': 'Test Pants', 'color': 'beige'}
    outfits = generate_outfit_combos(fixed, 'bottom', num_outfits=3)
    print(f'  Outfit Generation: {len(outfits)} outfits OK')
except Exception as e:
    print(f'  Outfit Generation ERROR: {e}')

try:
    from outfit_renderer import create_outfit_card
    if outfits:
        card = create_outfit_card(outfits[0], output_path='test_checkup.png')
        print(f'  Card Rendering: OK ({card.size})')
except Exception as e:
    print(f'  Card Rendering ERROR: {e}')

print('\n' + '='*60)
print('CHECK-UP COMPLETE')
print('='*60)
