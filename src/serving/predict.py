"""
WatchTower Inference Module.

Model loading and prediction logic.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger

from src.config.settings import settings


class ModelInference:
    """
    Model inference handler for WatchTower.
    
    Loads trained XGBoost model and provides prediction methods.
    """
    
    # Feature names in order (matching training data)
    FEATURE_NAMES = [
        "Destination Port", "Flow Duration", "Total Fwd Packets",
        "Total Length of Fwd Packets", "Fwd Packet Length Max",
        "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
        "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
        "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
        "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
        "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
        "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
        "Min Packet Length", "Max Packet Length", "Packet Length Mean",
        "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
        "PSH Flag Count", "ACK Flag Count", "Average Packet Size",
        "Subflow Fwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
        "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Max",
        "Active Min", "Idle Mean", "Idle Max", "Idle Min"
    ]
    
    # Class names
    CLASS_NAMES = [
        "Bots", "Brute Force", "DDoS", "DoS", 
        "Normal Traffic", "Port Scanning", "Web Attacks"
    ]
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize inference handler.
        
        Args:
            model_path: Path to saved model (default: models/xgboost_model.json)
        """
        self.model_path = model_path or settings.models_dir / "xgboost_model.json"
        self.model: Optional[xgb.XGBClassifier] = None
        self._is_loaded = False
    
    def load_model(self) -> bool:
        """
        Load the trained model from disk.
        
        Returns:
            True if loaded successfully
        """
        try:
            if not self.model_path.exists():
                logger.error(f"Model not found: {self.model_path}")
                return False
            
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(self.model_path))
            self._is_loaded = True
            
            logger.info(f"Model loaded from: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(self, features: Dict[str, float]) -> Tuple[str, int, float, Dict[str, float]]:
        """
        Make a single prediction.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Tuple of (class_name, class_id, confidence, probabilities)
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Convert to DataFrame with correct column order
        df = self._features_to_dataframe(features)
        
        # Predict
        prediction = self.model.predict(df)[0]
        probabilities = self.model.predict_proba(df)[0]
        
        # Get class name and confidence
        class_name = self.CLASS_NAMES[prediction] if prediction < len(self.CLASS_NAMES) else f"Class_{prediction}"
        confidence = float(probabilities[prediction])
        
        # Create probability dict
        proba_dict = {
            self.CLASS_NAMES[i]: float(p) 
            for i, p in enumerate(probabilities) 
            if i < len(self.CLASS_NAMES)
        }
        
        return class_name, int(prediction), confidence, proba_dict
    
    def predict_batch(self, samples: List[Dict[str, float]]) -> List[Tuple[str, int, float, Dict[str, float]]]:
        """
        Make batch predictions.
        
        Args:
            samples: List of feature dictionaries
            
        Returns:
            List of prediction tuples
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Convert all to DataFrame
        dfs = [self._features_to_dataframe(s) for s in samples]
        df = pd.concat(dfs, ignore_index=True)
        
        # Predict
        predictions = self.model.predict(df)
        probabilities = self.model.predict_proba(df)
        
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            class_name = self.CLASS_NAMES[pred] if pred < len(self.CLASS_NAMES) else f"Class_{pred}"
            confidence = float(proba[pred])
            proba_dict = {
                self.CLASS_NAMES[j]: float(p) 
                for j, p in enumerate(proba) 
                if j < len(self.CLASS_NAMES)
            }
            results.append((class_name, int(pred), confidence, proba_dict))
        
        return results
    
    def _features_to_dataframe(self, features: Dict[str, float]) -> pd.DataFrame:
        """
        Convert feature dict to DataFrame with correct column order.
        
        Args:
            features: Feature dictionary
            
        Returns:
            DataFrame with single row
        """
        # Map API field names to dataset column names
        field_mapping = {
            "destination_port": "Destination Port",
            "flow_duration": "Flow Duration",
            "total_fwd_packets": "Total Fwd Packets",
            "total_length_of_fwd_packets": "Total Length of Fwd Packets",
            "fwd_packet_length_max": "Fwd Packet Length Max",
            "fwd_packet_length_min": "Fwd Packet Length Min",
            "fwd_packet_length_mean": "Fwd Packet Length Mean",
            "fwd_packet_length_std": "Fwd Packet Length Std",
            "bwd_packet_length_max": "Bwd Packet Length Max",
            "bwd_packet_length_min": "Bwd Packet Length Min",
            "bwd_packet_length_mean": "Bwd Packet Length Mean",
            "bwd_packet_length_std": "Bwd Packet Length Std",
            "flow_bytes_per_s": "Flow Bytes/s",
            "flow_packets_per_s": "Flow Packets/s",
            "flow_iat_mean": "Flow IAT Mean",
            "flow_iat_std": "Flow IAT Std",
            "flow_iat_max": "Flow IAT Max",
            "flow_iat_min": "Flow IAT Min",
            "fwd_iat_total": "Fwd IAT Total",
            "fwd_iat_mean": "Fwd IAT Mean",
            "fwd_iat_std": "Fwd IAT Std",
            "fwd_iat_max": "Fwd IAT Max",
            "fwd_iat_min": "Fwd IAT Min",
            "bwd_iat_total": "Bwd IAT Total",
            "bwd_iat_mean": "Bwd IAT Mean",
            "bwd_iat_std": "Bwd IAT Std",
            "bwd_iat_max": "Bwd IAT Max",
            "bwd_iat_min": "Bwd IAT Min",
            "fwd_header_length": "Fwd Header Length",
            "bwd_header_length": "Bwd Header Length",
            "fwd_packets_per_s": "Fwd Packets/s",
            "bwd_packets_per_s": "Bwd Packets/s",
            "min_packet_length": "Min Packet Length",
            "max_packet_length": "Max Packet Length",
            "packet_length_mean": "Packet Length Mean",
            "packet_length_std": "Packet Length Std",
            "packet_length_variance": "Packet Length Variance",
            "fin_flag_count": "FIN Flag Count",
            "psh_flag_count": "PSH Flag Count",
            "ack_flag_count": "ACK Flag Count",
            "average_packet_size": "Average Packet Size",
            "subflow_fwd_bytes": "Subflow Fwd Bytes",
            "init_win_bytes_forward": "Init_Win_bytes_forward",
            "init_win_bytes_backward": "Init_Win_bytes_backward",
            "act_data_pkt_fwd": "act_data_pkt_fwd",
            "min_seg_size_forward": "min_seg_size_forward",
            "active_mean": "Active Mean",
            "active_max": "Active Max",
            "active_min": "Active Min",
            "idle_mean": "Idle Mean",
            "idle_max": "Idle Max",
            "idle_min": "Idle Min",
        }
        
        # Create row with correct column names
        row = {}
        for api_name, dataset_name in field_mapping.items():
            row[dataset_name] = features.get(api_name, 0.0)
        
        return pd.DataFrame([row])[self.FEATURE_NAMES]
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    @property
    def n_features(self) -> int:
        """Get number of features."""
        return len(self.FEATURE_NAMES)
    
    @property
    def n_classes(self) -> int:
        """Get number of classes."""
        return len(self.CLASS_NAMES)


# Global inference instance
inference = ModelInference()
