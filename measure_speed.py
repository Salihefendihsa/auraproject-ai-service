import requests
import time
import os

API_BASE = "http://localhost:8000"
API_KEY = "aura_2176c1c670b6338874c17472bec26c04f6b529bc33626a8c26292487e60e5dfdd"
IMAGE_PATH = r"c:\Users\SALİH\OneDrive\Desktop\AuraProject AI Service\test_image.jpg"

if not os.path.exists(IMAGE_PATH):
    # Try to find any jpg in the workspace
    IMAGE_PATH = r"c:\Users\SALİH\OneDrive\Desktop\AuraProject AI Service\ai_service\tests\test_images\sample.jpg"

def test_speed(turbo=False):
    print(f"\n[Testing] {'TURBO' if turbo else 'STANDARD'} Mode...")
    files = {'image': ('test.jpg', open(IMAGE_PATH, 'rb'), 'image/jpeg')}
    # Random note to bypass cache
    import random
    data = {'mode': 'single', 'turbo': str(turbo).lower(), 'user_note': f"Cache bypass {random.random()}"}
    headers = {'X-API-Key': API_KEY}
    
    start = time.time()
    try:
        # Use data=None and only pass files if testing multipart, 
        # but here we need both. requests handles this if you pass files and data.
        response = requests.post(
            f"{API_BASE}/ai/outfit", 
            files={'image': ('test.jpg', open(IMAGE_PATH, 'rb'), 'image/jpeg')}, 
            data=data, 
            headers=headers
        )
        end = time.time()
        
        if response.status_code == 200:
            duration = end - start
            res_json = response.json()
            proc_time = res_json.get("processing_time_ms", 0) / 1000
            print(f"✓ Success in {duration:.2f}s (Internal: {proc_time:.2f}s)")
            return duration
        else:
            print(f"✗ Failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

if __name__ == "__main__":
    # Warm up (segmenter load)
    test_speed(turbo=True)
    
    # Real test
    t1 = test_speed(turbo=False)
    t2 = test_speed(turbo=True)
    
    if t1 and t2:
        diff = t1 - t2
        pct = (diff / t1) * 100
        print(f"\n[RESULT] Turbo is {diff:.2f}s faster ({pct:.1f}% reduction)")
