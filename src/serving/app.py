"""
WatchTower FastAPI Application.

Real-time network intrusion detection API with Prometheus metrics.
"""

import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from loguru import logger

from src.config.settings import settings
from src.serving.schemas import (
    NetworkFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from src.serving.predict import inference


# =============================================================================
# Prometheus Metrics
# =============================================================================

PREDICTIONS_TOTAL = Counter(
    "watchtower_predictions_total",
    "Total number of predictions",
    ["prediction_class"]
)

PREDICTION_LATENCY = Histogram(
    "watchtower_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

REQUESTS_TOTAL = Counter(
    "watchtower_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)


# =============================================================================
# Lifespan Context Manager
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    logger.info("Starting WatchTower API...")
    
    # Load model
    if not inference.load_model():
        logger.warning("Model not loaded - predictions will fail until model is available")
    else:
        logger.info("Model loaded successfully!")
    
    yield
    
    logger.info("Shutting down WatchTower API...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="WatchTower API",
    description="Real-time Network Intrusion Detection API powered by XGBoost",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Middleware for metrics
# =============================================================================

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track request metrics."""
    start_time = time.time()
    response = await call_next(request)
    
    # Record request
    REQUESTS_TOTAL.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    return response


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "WatchTower API",
        "description": "Network Intrusion Detection with 99.877% accuracy",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if inference.is_loaded else "degraded",
        model_loaded=inference.is_loaded,
        version="1.0.0"
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information."""
    if not inference.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_name="WatchTower XGBoost",
        model_version="1.0.0",
        n_features=inference.n_features,
        n_classes=inference.n_classes,
        class_names=inference.CLASS_NAMES,
        accuracy=0.99877  # From our optimization
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: NetworkFeatures):
    """
    Predict attack type for network traffic features.
    
    Returns the predicted class, confidence score, and all class probabilities.
    """
    if not inference.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Convert to dict and predict
        feature_dict = features.model_dump()
        class_name, class_id, confidence, probabilities = inference.predict(feature_dict)
        
        # Record metrics
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        PREDICTIONS_TOTAL.labels(prediction_class=class_name).inc()
        
        return PredictionResponse(
            prediction=class_name,
            prediction_id=class_id,
            confidence=confidence,
            probabilities=probabilities
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction for multiple network traffic samples.
    
    More efficient than calling /predict multiple times.
    """
    if not inference.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.samples) == 0:
        raise HTTPException(status_code=400, detail="No samples provided")
    
    if len(request.samples) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 samples per batch")
    
    start_time = time.time()
    
    try:
        results = inference.predict_batch(request.samples)
        
        # Record metrics
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency / len(request.samples))  # Per-sample latency
        
        predictions = []
        for class_name, class_id, confidence, probabilities in results:
            PREDICTIONS_TOTAL.labels(prediction_class=class_name).inc()
            predictions.append(PredictionResponse(
                prediction=class_name,
                prediction_id=class_id,
                confidence=confidence,
                probabilities=probabilities
            ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_samples=len(predictions)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/classes", response_model=Dict[str, list])
async def get_classes():
    """Get list of attack classes."""
    return {"classes": inference.CLASS_NAMES}


# =============================================================================
# Main entry point
# =============================================================================

def main():
    """Run the API server."""
    import uvicorn
    
    uvicorn.run(
        "src.serving.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )


if __name__ == "__main__":
    main()
