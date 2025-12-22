"""Real system test - no f-strings with quotes"""
import requests

print('=== GERCEK SISTEM TESTI ===')
print()

# Test 1: Server
print('[1] Server Durumu')
try:
    resp = requests.get('http://localhost:8000/health', timeout=5)
    if resp.status_code == 200:
        print('    OK - Server calisiyor')
    else:
        print('    HATA - Status:', resp.status_code)
except Exception as e:
    print('    HATA -', str(e))

# Test 2: Catalog endpoint
print()
print('[2] /ai/outfit-catalog Endpoint')
try:
    from PIL import Image
    import io
    img = Image.new('RGB', (100, 100), color='gray')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    
    files = {'image': ('test.jpg', buf, 'image/jpeg')}
    data = {'item_category': 'bottom', 'item_color': 'beige', 'gender': 'male'}
    
    resp = requests.post('http://localhost:8000/ai/outfit-catalog', files=files, data=data, timeout=30)
    
    if resp.status_code == 200:
        result = resp.json()
        outfits = result.get('outfits', [])
        print('    OK -', len(outfits), 'outfit uretildi')
        
        if outfits:
            outfit = outfits[0]
            render_url = outfit.get('render_url', 'YOK')
            print('    Render URL:', render_url)
            
            if render_url and render_url != 'YOK':
                resp2 = requests.get('http://localhost:8000' + render_url, timeout=5)
                if resp2.status_code == 200:
                    print('    Gorsel: OK -', len(resp2.content), 'bytes')
                else:
                    print('    Gorsel: HATA -', resp2.status_code)
    else:
        print('    HATA - Status:', resp.status_code)
        print('    Mesaj:', resp.text[:300])
except Exception as e:
    print('    HATA -', str(e))

print()
print('=== TEST BITTI ===')
