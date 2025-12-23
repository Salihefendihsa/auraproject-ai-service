"""AURA System Health Check"""
print("=" * 60)
print("AURA SYSTEM HEALTH CHECK")
print("=" * 60)

errors = []
warnings = []

# 1. Core Imports
try:
    from ai_service.app.main import app
    print("[OK] FastAPI app loads")
except Exception as e:
    errors.append(f"FastAPI app: {e}")
    print(f"[FAIL] FastAPI app: {e}")

# 2. Routes
try:
    from ai_service.app.routes import router
    print("[OK] API routes load")
except Exception as e:
    errors.append(f"Routes: {e}")
    print(f"[FAIL] Routes: {e}")

# 3. Orchestrator
try:
    from ai_service.core.orchestrator import run_outfit_seed_job
    print("[OK] Orchestrator loads")
except Exception as e:
    errors.append(f"Orchestrator: {e}")
    print(f"[FAIL] Orchestrator: {e}")

# 4. Outfit Recommender
try:
    from ai_service.core.outfit_recommender import generate_outfits, load_catalog
    print("[OK] Outfit recommender loads")
except Exception as e:
    errors.append(f"Outfit recommender: {e}")
    print(f"[FAIL] Outfit recommender: {e}")

# 5. Catalog
try:
    from ai_service.core.outfit_recommender import load_catalog
    catalog = load_catalog()
    item_count = len(catalog.get("items", []))
    if item_count > 0:
        print(f"[OK] Catalog loaded: {item_count} items")
    else:
        warnings.append("Catalog is empty")
        print(f"[WARN] Catalog is empty")
except Exception as e:
    errors.append(f"Catalog: {e}")
    print(f"[FAIL] Catalog: {e}")

# 6. Lookbook Rules
try:
    from ai_service.core.outfit_recommender import load_lookbook_rules
    rules = load_lookbook_rules()
    print(f"[OK] Lookbook rules: {len(rules)} rules (Zara+H&M+Bershka)")
except Exception as e:
    errors.append(f"Lookbook: {e}")
    print(f"[FAIL] Lookbook: {e}")

# 7. Renderers
try:
    from ai_service.renderer.pose_estimation import estimate_pose
    from ai_service.renderer.human_parsing import parse_human_simple
    print("[OK] Renderers load (pose, parsing)")
except Exception as e:
    errors.append(f"Renderers: {e}")
    print(f"[FAIL] Renderers: {e}")

# 8. User Photo Tryon (NEW)
try:
    from ai_service.renderer.user_photo_detection import detect_user_photo
    from ai_service.renderer.user_photo_tryon import render_all_outfits_user_photo
    print("[OK] User photo tryon modules load")
except Exception as e:
    errors.append(f"User photo tryon: {e}")
    print(f"[FAIL] User photo tryon: {e}")

# 9. Cache
try:
    from ai_service.cache import cache_manager
    from ai_service.cache.seed_cache import get_seed_cache_stats
    stats = get_seed_cache_stats()
    enabled = stats.get("enabled", False)
    print(f"[OK] Cache system: enabled={enabled}")
except Exception as e:
    errors.append(f"Cache: {e}")
    print(f"[FAIL] Cache: {e}")

# 10. Config
try:
    from ai_service.config import get_settings
    from ai_service.config.settings import DEMO_BASELINE_VERSION
    settings = get_settings()
    print(f"[OK] Config loads (baseline={DEMO_BASELINE_VERSION})")
except Exception as e:
    errors.append(f"Config: {e}")
    print(f"[FAIL] Config: {e}")

# 11. Storage
try:
    from ai_service.core.storage import storage
    storage.ensure_directories()
    print(f"[OK] Storage directories ready")
except Exception as e:
    errors.append(f"Storage: {e}")
    print(f"[FAIL] Storage: {e}")

# 12. MongoDB (optional)
try:
    from ai_service.db import mongo
    connected = mongo.connect()
    if connected:
        print("[OK] MongoDB connected")
    else:
        warnings.append("MongoDB not connected (optional for demo)")
        print("[WARN] MongoDB not connected (optional)")
except Exception as e:
    warnings.append(f"MongoDB: {e}")
    print(f"[WARN] MongoDB: {e}")

# 13. Validation
try:
    from ai_service.core.validation import validate_outfit_seed_input, ALLOWED_MODES
    print(f"[OK] Validation ready (modes: {ALLOWED_MODES})")
except Exception as e:
    errors.append(f"Validation: {e}")
    print(f"[FAIL] Validation: {e}")

print("")
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Errors:   {len(errors)}")
print(f"Warnings: {len(warnings)}")
if len(errors) == 0:
    print("")
    print(">>> SISTEM CALISTIRILMAYA HAZIR <<<")
    print(">>> SYSTEM READY TO RUN <<<")
else:
    print("")
    print(">>> SISTEMDE HATALAR VAR <<<")
    for e in errors:
        print(f"  - {e}")
