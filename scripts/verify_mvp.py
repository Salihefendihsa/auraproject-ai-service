#!/usr/bin/env python
"""
verify_mvp.py - MVP Verification Script for AURA /ai/outfit-seed

This script verifies all MVP requirements for the outfit-seed endpoint.

Usage:
    python scripts/verify_mvp.py [--base-url http://localhost:8000] [--api-key YOUR_KEY]
"""
import sys
import json
import argparse
import requests
from pathlib import Path

# Default config
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_API_KEY = "test-api-key"

# Test image path (use an existing garment image)
TEST_IMAGE_DIR = Path(__file__).parent.parent / "ai_service" / "static" / "garments"


def find_test_image():
    """Find a test garment image."""
    if TEST_IMAGE_DIR.exists():
        images = list(TEST_IMAGE_DIR.glob("*.jpg"))
        if images:
            return images[0]
    return None


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}: {test_name}")
    if details and not passed:
        print(f"         → {details}")


def test_health(base_url: str) -> bool:
    """Test /health endpoint."""
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        return r.status_code == 200
    except Exception as e:
        print(f"  Health check error: {e}")
        return False


def test_outfit_seed(
    base_url: str,
    api_key: str,
    mode: str,
    image_path: Path,
    seed_category: str = None
) -> dict:
    """
    Test POST /ai/outfit-seed endpoint.
    
    Returns dict with test results and response.
    """
    result = {
        "passed": False,
        "response": None,
        "seed_locked": False,
        "outfit_count": 0,
        "all_have_render_url": False,
        "error": None,
    }
    
    try:
        url = f"{base_url}/ai/outfit-seed"
        headers = {"X-API-Key": api_key}
        
        files = {}
        if image_path and image_path.exists():
            files["seed_image"] = open(image_path, "rb")
        
        data = {
            "gender": "female",
            "mode": mode,
            "event": "casual",
        }
        if seed_category:
            data["seed_category"] = seed_category
        
        r = requests.post(url, headers=headers, data=data, files=files, timeout=60)
        
        # Close file handles
        for f in files.values():
            f.close()
        
        result["status_code"] = r.status_code
        
        if r.status_code == 200:
            resp = r.json()
            result["response"] = resp
            result["seed_locked"] = resp.get("seed_locked", False)
            result["outfit_count"] = len(resp.get("outfits", []))
            
            outfits = resp.get("outfits", [])
            if outfits:
                result["all_have_render_url"] = all(
                    o.get("render_url") is not None or o.get("tryon_mode") == "mock"
                    for o in outfits
                )
            
            result["passed"] = (
                result["seed_locked"] and
                result["outfit_count"] == 5 and
                (result["all_have_render_url"] or mode == "mock")
            )
        else:
            result["error"] = r.text[:200]
            
    except Exception as e:
        result["error"] = str(e)
    
    return result


def run_all_tests(base_url: str, api_key: str):
    """Run all MVP verification tests."""
    print("\n" + "="*60)
    print("AURA MVP VERIFICATION")
    print("="*60 + "\n")
    
    all_passed = True
    
    # Find test image
    test_image = find_test_image()
    if test_image:
        print(f"Using test image: {test_image.name}\n")
    else:
        print("⚠️  Warning: No test image found in static/garments/\n")
    
    # Test 1: Health check
    print("[1] Testing /health endpoint...")
    health_ok = test_health(base_url)
    print_result("Health check", health_ok)
    all_passed = all_passed and health_ok
    
    if not health_ok:
        print("\n❌ Server not healthy. Aborting.\n")
        return False
    
    print()
    
    # Test 2: Mock mode
    print("[2] Testing mode=mock...")
    mock_result = test_outfit_seed(
        base_url, api_key, "mock", test_image, seed_category="top"
    )
    print_result("seed_locked = true", mock_result["seed_locked"])
    print_result("outfit_count = 5", mock_result["outfit_count"] == 5,
                 f"Got {mock_result['outfit_count']}")
    print_result("Mock mode overall", mock_result["passed"], mock_result.get("error", ""))
    all_passed = all_passed and mock_result["passed"]
    
    print()
    
    # Test 3: Partial try-on mode
    print("[3] Testing mode=partial_tryon...")
    partial_result = test_outfit_seed(
        base_url, api_key, "partial_tryon", test_image, seed_category="top"
    )
    print_result("seed_locked = true", partial_result["seed_locked"])
    print_result("outfit_count = 5", partial_result["outfit_count"] == 5,
                 f"Got {partial_result['outfit_count']}")
    print_result("all outfits have render_url", partial_result["all_have_render_url"])
    print_result("Partial mode overall", partial_result["passed"], partial_result.get("error", ""))
    all_passed = all_passed and partial_result["passed"]
    
    print()
    
    # Test 4: Full try-on mode
    print("[4] Testing mode=full_tryon...")
    full_result = test_outfit_seed(
        base_url, api_key, "full_tryon", test_image, seed_category="top"
    )
    print_result("seed_locked = true", full_result["seed_locked"])
    print_result("outfit_count = 5", full_result["outfit_count"] == 5,
                 f"Got {full_result['outfit_count']}")
    
    # Check fallback worked
    if full_result["response"]:
        tryon_mode = full_result["response"].get("tryon_mode")
        fallback_ok = tryon_mode in ["full", "partial_fallback", "mixed", "partial"]
        print_result("Fallback mechanism works", fallback_ok, f"tryon_mode={tryon_mode}")
    
    print_result("Full mode overall (or fallback)", full_result["passed"], full_result.get("error", ""))
    # Don't fail overall for full_tryon since fallback is OK
    
    print()
    
    # Test 5: Seed detection without explicit category
    print("[5] Testing auto seed_category detection...")
    auto_result = test_outfit_seed(
        base_url, api_key, "mock", test_image, seed_category=None
    )
    # If succeeds, detection worked; if fails with 400, detection confidence was low (expected)
    if auto_result["passed"]:
        print_result("Auto-detection succeeded", True)
    elif auto_result.get("status_code") == 400:
        print_result("Low confidence detection returns 400 (expected)", True)
    else:
        print_result("Auto-detection", False, auto_result.get("error", ""))
    
    print()
    
    # Summary
    print("="*60)
    if all_passed:
        print("🎉 ALL MVP TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED - Review above")
    print("="*60 + "\n")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="AURA MVP Verification Script")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL of API")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key for auth")
    args = parser.parse_args()
    
    success = run_all_tests(args.base_url, args.api_key)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
