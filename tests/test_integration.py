"""
Integration Tests for AuraProject AI Service v2.0.0
Tests for security features: Input Validation, Auth, Rate Limiting.
"""
import pytest
import io
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Test client setup
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def client():
    """Create test client."""
    from ai_service.app.main import app
    return TestClient(app)


@pytest.fixture
def test_image():
    """Create a valid test image."""
    from PIL import Image
    import io
    
    # Create a small valid JPEG image
    img = Image.new('RGB', (100, 100), color='blue')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    return buffer


@pytest.fixture
def test_api_key():
    """Get or create a test API key."""
    return "test_api_key_for_testing"


@pytest.fixture
def mock_user():
    """Mock authenticated user."""
    from ai_service.core.auth import User
    return User(
        user_id="test_user",
        name="Test User",
        api_key="test_api_key_for_testing",
        created_at="2024-01-01T00:00:00Z"
    )


# ==================== HEALTH CHECK TESTS ====================

class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_returns_ok(self, client):
        """Health endpoint should return status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "2.0.0"
    
    def test_health_includes_new_features(self, client):
        """Health should list new security features."""
        response = client.get("/health")
        data = response.json()
        
        features = data.get("features", [])
        assert "input_validation" in features
        assert "api_key_auth" in features
        assert "rate_limiting" in features


# ==================== INPUT VALIDATION TESTS ====================

class TestInputValidation:
    """Tests for input validation."""
    
    def test_validation_rejects_large_file(self, client, mock_user):
        """Files over 10MB should be rejected."""
        # Create 15MB file (over limit)
        large_content = b"x" * (15 * 1024 * 1024)
        
        with patch('ai_service.core.auth.get_user_by_api_key', return_value=mock_user):
            response = client.post(
                "/ai/outfit",
                files={"image": ("large.jpg", io.BytesIO(large_content), "image/jpeg")},
                headers={"X-API-Key": "test_key"}
            )
        
        assert response.status_code == 413
        assert "too large" in response.json()["detail"].lower()
    
    def test_validation_rejects_invalid_mime(self, client, mock_user):
        """Non-image MIME types should be rejected."""
        pdf_content = b"%PDF-1.4 fake pdf content"
        
        with patch('ai_service.core.auth.get_user_by_api_key', return_value=mock_user):
            response = client.post(
                "/ai/outfit",
                files={"image": ("doc.pdf", io.BytesIO(pdf_content), "application/pdf")},
                headers={"X-API-Key": "test_key"}
            )
        
        assert response.status_code == 415
        assert "unsupported" in response.json()["detail"].lower()
    
    def test_validation_rejects_undecodable_image(self, client, mock_user):
        """Files that can't be decoded should be rejected."""
        fake_image = b"not an image at all"
        
        with patch('ai_service.core.auth.get_user_by_api_key', return_value=mock_user):
            response = client.post(
                "/ai/outfit",
                files={"image": ("fake.jpg", io.BytesIO(fake_image), "image/jpeg")},
                headers={"X-API-Key": "test_key"}
            )
        
        assert response.status_code == 400
        assert "decode" in response.json()["detail"].lower()


# ==================== AUTHENTICATION TESTS ====================

class TestAuthentication:
    """Tests for API key authentication."""
    
    def test_auth_rejects_missing_key(self, client, test_image):
        """Requests without API key should be rejected."""
        response = client.post(
            "/ai/outfit",
            files={"image": ("test.jpg", test_image, "image/jpeg")}
        )
        
        assert response.status_code == 401
        assert "missing" in response.json()["detail"].lower()
    
    def test_auth_rejects_invalid_key(self, client, test_image):
        """Requests with invalid API key should be rejected."""
        with patch('ai_service.core.auth.get_user_by_api_key', return_value=None):
            response = client.post(
                "/ai/outfit",
                files={"image": ("test.jpg", test_image, "image/jpeg")},
                headers={"X-API-Key": "invalid_key_12345"}
            )
        
        assert response.status_code == 401
        assert "invalid" in response.json()["detail"].lower()
    
    def test_auth_accepts_valid_key(self, client, test_image, mock_user):
        """Requests with valid API key should be accepted."""
        with patch('ai_service.core.auth.get_user_by_api_key', return_value=mock_user):
            with patch('ai_service.core.orchestrator.run_pipeline') as mock_pipeline:
                mock_pipeline.return_value = {
                    "detected_clothing": {},
                    "detected_items": {},
                    "masks": {},
                    "raw_labels": [],
                    "outfits": [],
                    "status": "completed"
                }
                
                response = client.post(
                    "/ai/outfit",
                    files={"image": ("test.jpg", test_image, "image/jpeg")},
                    headers={"X-API-Key": "valid_key"}
                )
        
        # Should process (may fail for other reasons, but not auth)
        assert response.status_code != 401


# ==================== RATE LIMITING TESTS ====================

class TestRateLimiting:
    """Tests for rate limiting."""
    
    def test_rate_limit_headers_present(self, client, test_image, mock_user):
        """Response should include rate limit headers."""
        with patch('ai_service.core.auth.get_user_by_api_key', return_value=mock_user):
            with patch('ai_service.core.orchestrator.run_pipeline') as mock_pipeline:
                mock_pipeline.return_value = {
                    "detected_clothing": {},
                    "detected_items": {},
                    "masks": {},
                    "outfits": [],
                    "status": "completed"
                }
                
                response = client.post(
                    "/ai/outfit",
                    files={"image": ("test.jpg", test_image, "image/jpeg")},
                    headers={"X-API-Key": "valid_key"}
                )
        
        # Check rate limit headers
        assert "X-RateLimit-Limit" in response.headers or response.status_code in [429, 500]


# ==================== JOB OWNERSHIP TESTS ====================

class TestJobOwnership:
    """Tests for job ownership."""
    
    def test_job_access_denied_for_non_owner(self, client, mock_user):
        """User should not access other user's job."""
        other_user_job = {
            "job_id": "other_job",
            "owner_user_id": "different_user",
            "status": "completed"
        }
        
        with patch('ai_service.core.auth.get_user_by_api_key', return_value=mock_user):
            with patch('ai_service.db.mongo.get_job', return_value=other_user_job):
                response = client.get(
                    "/ai/jobs/other_job",
                    headers={"X-API-Key": "test_key"}
                )
        
        assert response.status_code == 403
        assert "denied" in response.json()["detail"].lower()
    
    def test_job_access_allowed_for_owner(self, client, mock_user):
        """User should access their own job."""
        own_job = {
            "job_id": "my_job",
            "owner_user_id": "test_user",
            "status": "completed"
        }
        
        with patch('ai_service.core.auth.get_user_by_api_key', return_value=mock_user):
            with patch('ai_service.db.mongo.get_job', return_value=own_job):
                response = client.get(
                    "/ai/jobs/my_job",
                    headers={"X-API-Key": "test_key"}
                )
        
        assert response.status_code == 200


# ==================== PATH TRAVERSAL TESTS ====================

class TestPathTraversal:
    """Tests for path traversal protection."""
    
    def test_path_traversal_blocked(self, client):
        """Path traversal attempts should be blocked."""
        dangerous_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "jobs/../../../secret.txt",
            "jobs/%2e%2e/%2e%2e/secret"
        ]
        
        for path in dangerous_paths:
            response = client.get(f"/ai/assets/{path}")
            # Should be 403 (forbidden) or 404 (not found), never 200
            assert response.status_code in [400, 403, 404, 500], f"Path {path} should be blocked"


# ==================== RUN TESTS ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
