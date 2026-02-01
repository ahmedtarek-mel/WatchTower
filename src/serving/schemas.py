"""
WatchTower API Schemas.

Pydantic models for request/response validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class NetworkFeatures(BaseModel):
    """Input features for network traffic prediction."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "destination_port": 80,
                "flow_duration": 1234567,
                "total_fwd_packets": 10,
                "total_length_of_fwd_packets": 1500,
                "fwd_packet_length_max": 1460,
                "fwd_packet_length_min": 40,
                "fwd_packet_length_mean": 500.5,
                "fwd_packet_length_std": 200.3,
            }
        }
    )
    
    destination_port: float = Field(..., description="Destination port number")
    flow_duration: float = Field(..., description="Flow duration in microseconds")
    total_fwd_packets: float = Field(..., description="Total forward packets")
    total_length_of_fwd_packets: float = Field(..., description="Total length of forward packets")
    fwd_packet_length_max: float = Field(..., description="Max forward packet length")
    fwd_packet_length_min: float = Field(..., description="Min forward packet length")
    fwd_packet_length_mean: float = Field(..., description="Mean forward packet length")
    fwd_packet_length_std: float = Field(..., description="Std of forward packet length")
    bwd_packet_length_max: float = Field(default=0, description="Max backward packet length")
    bwd_packet_length_min: float = Field(default=0, description="Min backward packet length")
    bwd_packet_length_mean: float = Field(default=0, description="Mean backward packet length")
    bwd_packet_length_std: float = Field(default=0, description="Std of backward packet length")
    flow_bytes_per_s: float = Field(default=0, description="Flow bytes per second")
    flow_packets_per_s: float = Field(default=0, description="Flow packets per second")
    flow_iat_mean: float = Field(default=0, description="Flow IAT mean")
    flow_iat_std: float = Field(default=0, description="Flow IAT std")
    flow_iat_max: float = Field(default=0, description="Flow IAT max")
    flow_iat_min: float = Field(default=0, description="Flow IAT min")
    fwd_iat_total: float = Field(default=0, description="Forward IAT total")
    fwd_iat_mean: float = Field(default=0, description="Forward IAT mean")
    fwd_iat_std: float = Field(default=0, description="Forward IAT std")
    fwd_iat_max: float = Field(default=0, description="Forward IAT max")
    fwd_iat_min: float = Field(default=0, description="Forward IAT min")
    bwd_iat_total: float = Field(default=0, description="Backward IAT total")
    bwd_iat_mean: float = Field(default=0, description="Backward IAT mean")
    bwd_iat_std: float = Field(default=0, description="Backward IAT std")
    bwd_iat_max: float = Field(default=0, description="Backward IAT max")
    bwd_iat_min: float = Field(default=0, description="Backward IAT min")
    fwd_header_length: float = Field(default=0, description="Forward header length")
    bwd_header_length: float = Field(default=0, description="Backward header length")
    fwd_packets_per_s: float = Field(default=0, description="Forward packets per second")
    bwd_packets_per_s: float = Field(default=0, description="Backward packets per second")
    min_packet_length: float = Field(default=0, description="Min packet length")
    max_packet_length: float = Field(default=0, description="Max packet length")
    packet_length_mean: float = Field(default=0, description="Packet length mean")
    packet_length_std: float = Field(default=0, description="Packet length std")
    packet_length_variance: float = Field(default=0, description="Packet length variance")
    fin_flag_count: float = Field(default=0, description="FIN flag count")
    psh_flag_count: float = Field(default=0, description="PSH flag count")
    ack_flag_count: float = Field(default=0, description="ACK flag count")
    average_packet_size: float = Field(default=0, description="Average packet size")
    subflow_fwd_bytes: float = Field(default=0, description="Subflow forward bytes")
    init_win_bytes_forward: float = Field(default=0, description="Init window bytes forward")
    init_win_bytes_backward: float = Field(default=0, description="Init window bytes backward")
    act_data_pkt_fwd: float = Field(default=0, description="Active data packets forward")
    min_seg_size_forward: float = Field(default=0, description="Min segment size forward")
    active_mean: float = Field(default=0, description="Active mean")
    active_max: float = Field(default=0, description="Active max")
    active_min: float = Field(default=0, description="Active min")
    idle_mean: float = Field(default=0, description="Idle mean")
    idle_max: float = Field(default=0, description="Idle max")
    idle_min: float = Field(default=0, description="Idle min")


class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    
    prediction: str = Field(..., description="Predicted attack type")
    prediction_id: int = Field(..., description="Numeric prediction class")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    
    samples: List[Dict[str, float]] = Field(..., description="List of feature dictionaries")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_samples: int = Field(..., description="Total samples processed")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    n_features: int = Field(..., description="Number of features")
    n_classes: int = Field(..., description="Number of classes")
    class_names: List[str] = Field(..., description="Class names")
    accuracy: Optional[float] = Field(None, description="Model accuracy")
