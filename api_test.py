"""AURA API Test"""
import requests
import os
from PIL import Image

# Create a test image
print("Creating test image...")
img = Image.new('RGB', (200, 300), color='blue')
test_image = 'test_seed.jpg'
img.save(test_image)
print(f"Test image: {test_image}")

# Test API
print("Testing /ai/outfit-seed endpoint...")
try:
    with open(test_image, 'rb') as f:
        response = requests.post(
            'http://localhost:8000/ai/outfit-seed',
            files={'seed_image': f},
            data={'gender': 'female', 'mode': 'mock', 'seed_category': 'top'}
        )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        job_id = data.get("job_id", "N/A")
        seed_cat = data.get("seed", {}).get("category", "N/A")
        outfit_count = len(data.get("outfits", []))
        baseline = data.get("baseline_version", "N/A")
        
        print(f"Job ID: {job_id}")
        print(f"Seed Category: {seed_cat}")
        print(f"Outfits: {outfit_count}")
        print(f"Baseline: {baseline}")
        print("")
        print(">>> API TEST: SUCCESS <<<")
    else:
        print(f"Error: {response.text[:500]}")
        print(">>> API TEST: FAILED <<<")
except Exception as e:
    print(f"Error: {e}")
    print(">>> API TEST: FAILED <<<")

# Cleanup
if os.path.exists(test_image):
    os.remove(test_image)
