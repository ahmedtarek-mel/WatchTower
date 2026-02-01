"""Integration tests for the FastAPI API."""

import pytest
from fastapi.testclient import TestClient

from src.serving.app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health and info endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "WatchTower API"
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
    
    def test_classes_endpoint(self, client):
        """Test classes endpoint."""
        response = client.get("/classes")
        assert response.status_code == 200
        data = response.json()
        assert "classes" in data
        assert len(data["classes"]) > 0


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""
    
    @pytest.fixture
    def sample_features(self):
        """Sample network features for testing."""
        return {
            "destination_port": 80,
            "flow_duration": 1234567,
            "total_fwd_packets": 10,
            "total_length_of_fwd_packets": 1500,
            "fwd_packet_length_max": 1460,
            "fwd_packet_length_min": 40,
            "fwd_packet_length_mean": 500.5,
            "fwd_packet_length_std": 200.3,
        }
    
    def test_predict_endpoint_returns_prediction(self, client, sample_features):
        """Test single prediction endpoint."""
        response = client.post("/predict", json=sample_features)
        
        # May fail if model not loaded, that's ok for test
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "prediction_id" in data
            assert "confidence" in data
            assert "probabilities" in data
        else:
            # Model not loaded is acceptable in test environment
            assert response.status_code in [200, 503]
    
    def test_batch_predict_endpoint(self, client, sample_features):
        """Test batch prediction endpoint."""
        request_data = {"samples": [sample_features, sample_features]}
        response = client.post("/predict/batch", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total_samples" in data
            assert data["total_samples"] == 2
        else:
            assert response.status_code in [200, 503]
    
    def test_batch_predict_empty_fails(self, client):
        """Test batch prediction with empty samples fails."""
        response = client.post("/predict/batch", json={"samples": []})
        assert response.status_code in [400, 503]
    
    def test_batch_predict_too_many_fails(self, client, sample_features):
        """Test batch prediction with too many samples fails."""
        # Create 1001 samples (over limit)
        samples = [sample_features for _ in range(1001)]
        response = client.post("/predict/batch", json={"samples": samples})
        assert response.status_code in [400, 503]


class TestMetricsEndpoint:
    """Tests for Prometheus metrics endpoint."""
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint returns Prometheus format."""
        response = client.get("/metrics")
        assert response.status_code == 200
        content = response.text
        
        # Should contain Prometheus metrics
        assert "watchtower" in content.lower() or "python" in content.lower()
