"""Tests for serving module components."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.serving.schemas import (
    NetworkFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from src.serving.predict import ModelInference


class TestNetworkFeaturesSchema:
    """Tests for NetworkFeatures Pydantic model."""
    
    def test_create_with_required_fields(self):
        """Test creating with required fields only."""
        features = NetworkFeatures(
            destination_port=80,
            flow_duration=1000,
            total_fwd_packets=5,
            total_length_of_fwd_packets=500,
            fwd_packet_length_max=100,
            fwd_packet_length_min=50,
            fwd_packet_length_mean=75,
            fwd_packet_length_std=10,
        )
        
        assert features.destination_port == 80
        assert features.flow_duration == 1000
    
    def test_optional_fields_default_to_zero(self):
        """Test optional fields have default values."""
        features = NetworkFeatures(
            destination_port=80,
            flow_duration=1000,
            total_fwd_packets=5,
            total_length_of_fwd_packets=500,
            fwd_packet_length_max=100,
            fwd_packet_length_min=50,
            fwd_packet_length_mean=75,
            fwd_packet_length_std=10,
        )
        
        assert features.bwd_packet_length_max == 0
        assert features.flow_iat_mean == 0
    
    def test_model_dump(self):
        """Test model can be dumped to dict."""
        features = NetworkFeatures(
            destination_port=80,
            flow_duration=1000,
            total_fwd_packets=5,
            total_length_of_fwd_packets=500,
            fwd_packet_length_max=100,
            fwd_packet_length_min=50,
            fwd_packet_length_mean=75,
            fwd_packet_length_std=10,
        )
        
        data = features.model_dump()
        
        assert isinstance(data, dict)
        assert "destination_port" in data


class TestPredictionResponseSchema:
    """Tests for PredictionResponse Pydantic model."""
    
    def test_create_response(self):
        """Test creating a prediction response."""
        response = PredictionResponse(
            prediction="Normal Traffic",
            prediction_id=4,
            confidence=0.95,
            probabilities={"Normal Traffic": 0.95, "DDoS": 0.05},
        )
        
        assert response.prediction == "Normal Traffic"
        assert response.confidence == 0.95
    
    def test_confidence_bounds(self):
        """Test confidence must be between 0 and 1."""
        # Valid confidence
        response = PredictionResponse(
            prediction="Test",
            prediction_id=0,
            confidence=0.5,
            probabilities={},
        )
        assert response.confidence == 0.5


class TestHealthResponseSchema:
    """Tests for HealthResponse Pydantic model."""
    
    def test_create_health_response(self):
        """Test creating health response."""
        response = HealthResponse(
            status="healthy",
            model_loaded=True,
            version="1.0.0",
        )
        
        assert response.status == "healthy"
        assert response.model_loaded is True


class TestModelInfoResponseSchema:
    """Tests for ModelInfoResponse Pydantic model."""
    
    def test_create_model_info(self):
        """Test creating model info response."""
        response = ModelInfoResponse(
            model_name="XGBoost",
            model_version="1.0",
            n_features=52,
            n_classes=7,
            class_names=["Normal", "DDoS"],
        )
        
        assert response.n_features == 52
        assert len(response.class_names) == 2


class TestModelInference:
    """Tests for ModelInference class."""
    
    @pytest.fixture
    def inference(self):
        """Create inference instance."""
        return ModelInference()
    
    def test_feature_names_defined(self, inference):
        """Test that feature names are defined."""
        assert len(inference.FEATURE_NAMES) > 0
        assert "Destination Port" in inference.FEATURE_NAMES
    
    def test_class_names_defined(self, inference):
        """Test that class names are defined."""
        assert len(inference.CLASS_NAMES) > 0
        assert "Normal Traffic" in inference.CLASS_NAMES
    
    def test_n_features_property(self, inference):
        """Test n_features property."""
        assert inference.n_features == len(inference.FEATURE_NAMES)
    
    def test_n_classes_property(self, inference):
        """Test n_classes property."""
        assert inference.n_classes == len(inference.CLASS_NAMES)
    
    def test_is_loaded_initially_false(self, inference):
        """Test model is not loaded initially."""
        assert inference.is_loaded is False
    
    def test_predict_raises_when_not_loaded(self, inference):
        """Test predict raises error when model not loaded."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            inference.predict({"destination_port": 80})
    
    def test_features_to_dataframe(self, inference):
        """Test feature dict to DataFrame conversion."""
        features = {
            "destination_port": 80,
            "flow_duration": 1000,
        }
        
        # Access private method for testing
        df = inference._features_to_dataframe(features)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "Destination Port" in df.columns


class TestBatchPredictionRequest:
    """Tests for BatchPredictionRequest schema."""
    
    def test_create_batch_request(self):
        """Test creating batch request."""
        request = BatchPredictionRequest(
            samples=[
                {"destination_port": 80},
                {"destination_port": 443},
            ]
        )
        
        assert len(request.samples) == 2
    
    def test_empty_samples_allowed(self):
        """Test empty samples list is technically allowed by schema."""
        request = BatchPredictionRequest(samples=[])
        assert len(request.samples) == 0
