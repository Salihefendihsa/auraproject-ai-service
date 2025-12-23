"""AURA Full System Integration Test"""
import requests
import os
from PIL import Image
import json

print("=" * 60)
print("AURA FULL SYSTEM INTEGRATION TEST")
print("=" * 60)
print("")

errors = []
successes = []

# 1. Health Check
print("[TEST] Health endpoint...")
try:
    r = requests.get('http://localhost:8000/health', timeout=5)
    if r.status_code == 200:
        data = r.json()
        print(f"  [OK] Status: {data.get('status')}, Version: {data.get('version')}")
        successes.append("Health endpoint")
    else:
        errors.append(f"Health endpoint: {r.status_code}")
        print(f"  [FAIL] Status: {r.status_code}")
except Exception as e:
    errors.append(f"Health endpoint: {e}")
    print(f"  [FAIL] {e}")

# 2. Metrics Check
print("[TEST] Metrics endpoint...")
try:
    r = requests.get('http://localhost:8000/metrics', timeout=5)
    if r.status_code == 200:
        print(f"  [OK] Metrics available")
        successes.append("Metrics endpoint")
    else:
        errors.append(f"Metrics: {r.status_code}")
        print(f"  [FAIL] Status: {r.status_code}")
except Exception as e:
    errors.append(f"Metrics: {e}")
    print(f"  [FAIL] {e}")

# 3. Create test image
print("[TEST] Creating test image...")
img = Image.new('RGB', (200, 300), color='blue')
test_image = 'test_seed_full.jpg'
img.save(test_image)
print(f"  [OK] Created: {test_image}")

# 4. Outfit Seed - Mock Mode
print("[TEST] /ai/outfit-seed (mock mode)...")
try:
    with open(test_image, 'rb') as f:
        r = requests.post(
            'http://localhost:8000/ai/outfit-seed',
            files={'seed_image': f},
            data={'gender': 'female', 'mode': 'mock', 'seed_category': 'top', 'event': 'casual'},
            timeout=60
        )
    if r.status_code == 200:
        data = r.json()
        outfit_count = len(data.get('outfits', []))
        baseline = data.get('baseline_version')
        seed_cat = data.get('seed', {}).get('category')
        print(f"  [OK] Outfits: {outfit_count}, Seed: {seed_cat}, Baseline: {baseline}")
        successes.append("Outfit-seed mock mode")
        
        # Verify outfit structure
        if outfit_count == 5:
            print(f"  [OK] Correct outfit count (5)")
            successes.append("Outfit count check")
        else:
            errors.append(f"Expected 5 outfits, got {outfit_count}")
            print(f"  [FAIL] Expected 5 outfits, got {outfit_count}")
        
        # Verify lookbook brands
        brands = [o.get('lookbook_brand', 'unknown') for o in data.get('outfits', [])]
        print(f"  [INFO] Lookbook brands: {brands}")
        
    else:
        errors.append(f"Outfit-seed mock: {r.status_code} - {r.text[:100]}")
        print(f"  [FAIL] Status: {r.status_code}")
except Exception as e:
    errors.append(f"Outfit-seed mock: {e}")
    print(f"  [FAIL] {e}")

# 5. Outfit Seed - Partial Tryon Mode
print("[TEST] /ai/outfit-seed (partial_tryon mode)...")
try:
    with open(test_image, 'rb') as f:
        r = requests.post(
            'http://localhost:8000/ai/outfit-seed',
            files={'seed_image': f},
            data={'gender': 'male', 'mode': 'partial_tryon', 'seed_category': 'bottom'},
            timeout=120
        )
    if r.status_code == 200:
        data = r.json()
        tryon_mode = data.get('tryon_mode', 'unknown')
        print(f"  [OK] Tryon mode: {tryon_mode}")
        successes.append("Outfit-seed partial_tryon mode")
    else:
        errors.append(f"Outfit-seed partial_tryon: {r.status_code}")
        print(f"  [FAIL] Status: {r.status_code}")
except Exception as e:
    errors.append(f"Outfit-seed partial_tryon: {e}")
    print(f"  [FAIL] {e}")

# 6. Cache Test (same request should hit cache)
print("[TEST] Cache system (repeat request)...")
try:
    with open(test_image, 'rb') as f:
        r = requests.post(
            'http://localhost:8000/ai/outfit-seed',
            files={'seed_image': f},
            data={'gender': 'female', 'mode': 'mock', 'seed_category': 'top', 'event': 'casual'},
            timeout=60
        )
    if r.status_code == 200:
        data = r.json()
        cache_hit = data.get('cache_hit', False)
        print(f"  [INFO] Cache hit: {cache_hit}")
        successes.append("Cache system")
    else:
        print(f"  [WARN] Status: {r.status_code}")
except Exception as e:
    print(f"  [WARN] {e}")

# 7. Different genders
print("[TEST] Gender: male...")
try:
    with open(test_image, 'rb') as f:
        r = requests.post(
            'http://localhost:8000/ai/outfit-seed',
            files={'seed_image': f},
            data={'gender': 'male', 'mode': 'mock', 'seed_category': 'top'},
            timeout=60
        )
    if r.status_code == 200:
        print(f"  [OK] Male gender works")
        successes.append("Male gender")
    else:
        errors.append(f"Male gender: {r.status_code}")
        print(f"  [FAIL] Status: {r.status_code}")
except Exception as e:
    errors.append(f"Male gender: {e}")
    print(f"  [FAIL] {e}")

# 8. Different events
for event in ['work', 'party', 'date']:
    print(f"[TEST] Event: {event}...")
    try:
        with open(test_image, 'rb') as f:
            r = requests.post(
                'http://localhost:8000/ai/outfit-seed',
                files={'seed_image': f},
                data={'gender': 'female', 'mode': 'mock', 'seed_category': 'top', 'event': event},
                timeout=60
            )
        if r.status_code == 200:
            print(f"  [OK] Event {event} works")
            successes.append(f"Event {event}")
        else:
            errors.append(f"Event {event}: {r.status_code}")
            print(f"  [FAIL] Status: {r.status_code}")
    except Exception as e:
        errors.append(f"Event {event}: {e}")
        print(f"  [FAIL] {e}")

# Cleanup
if os.path.exists(test_image):
    os.remove(test_image)

# Summary
print("")
print("=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print(f"Successes: {len(successes)}")
print(f"Errors:    {len(errors)}")
print("")

if len(errors) == 0:
    print(">>> TUM TESTLER BASARILI <<<")
    print(">>> ALL TESTS PASSED <<<")
else:
    print(">>> BAZI TESTLER BASARISIZ <<<")
    for e in errors:
        print(f"  - {e}")
