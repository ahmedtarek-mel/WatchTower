# Model Card: WatchTower Network Intrusion Detection

## Model Details

### Overview
- **Model Name**: WatchTower XGBoost Classifier
- **Version**: 1.0.0
- **Type**: Multi-class Classification
- **Framework**: XGBoost 2.0+
- **License**: MIT

### Developers
- **Primary Developer**: Ahmed Tarek
- **Role**: Data Scientist & Machine Learning Engineer
- **Contact**: ahmedtarekmel@gmail.com
- **LinkedIn**: [linkedin.com/in/ahmed-tarek-mel](https://www.linkedin.com/in/ahmed-tarek-mel)
- **Portfolio**: [datascienceportfol.io/AhmedTarek](https://www.datascienceportfol.io/AhmedTarek)

### Model Date
- **Training Date**: January 2026
- **Last Updated**: January 2026

---

## Intended Use

### Primary Use Cases
- Real-time network traffic classification
- Intrusion Detection System (IDS) backend
- Security Operations Center (SOC) alerting
- Network traffic analysis and monitoring

### Intended Users
- Security analysts and researchers
- Network administrators
- SOC teams
- Cybersecurity professionals

### Out-of-Scope Uses
- This model should NOT be used for:
  - Making autonomous blocking decisions without human review
  - Legal evidence without expert analysis
  - Classification of encrypted traffic payloads
  - Zero-day attack detection (only detects known patterns)

---

## Training Data

### Dataset
- **Name**: CICIDS2017 (Canadian Institute for Cybersecurity)
- **Source**: [University of New Brunswick](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Size**: 2.8+ million samples
- **Features**: 78 network flow features

### Data Collection
- Captured over 5 days in July 2017
- Contains both benign traffic and attack scenarios
- Attacks were executed in a controlled lab environment

### Class Distribution

| Class | Description | Approx. % |
|-------|-------------|-----------|
| Normal Traffic | Benign network activity | 83% |
| DoS | Denial of Service attacks | 8% |
| DDoS | Distributed DoS attacks | 4% |
| Port Scanning | Network reconnaissance | 2% |
| Brute Force | Password/authentication attacks | 1% |
| Web Attacks | SQL injection, XSS, etc. | 1% |
| Bots | Botnet activity | 1% |

### Preprocessing
- Removed infinite and NaN values
- Standardized features using StandardScaler
- Label encoded target classes
- Stratified train/validation/test split (70/15/15)

---

## Model Architecture

### Algorithm
- **Type**: Gradient Boosted Decision Trees (XGBoost)
- **Tree Method**: Histogram-based (`hist`)
- **GPU Acceleration**: CUDA support via `device='cuda'`

### Hyperparameters (Optimized via Optuna)

| Parameter | Value |
|-----------|-------|
| n_estimators | 377 |
| max_depth | 6 |
| learning_rate | 0.296 |
| tree_method | hist |
| eval_metric | mlogloss |

### Training Configuration
- **Optimization**: 30 Optuna trials with Bayesian search
- **Early Stopping**: 10 rounds (validation monitoring)
- **Hardware**: NVIDIA GPU with CUDA

---

## Evaluation Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 99.877% |
| **F1 Macro** | 95.9% |
| **F1 Weighted** | 99.88% |
| **Precision Macro** | 96.3% |
| **Recall Macro** | 95.6% |

### Benchmark Comparison

| Model | Accuracy | Source |
|-------|----------|--------|
| **WatchTower (Ours)** | 99.877% | This work |
| XGBoost (Literature SOTA) | 99.89% | Published research |
| Random Forest | 97.10% | Baseline |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal Traffic | 99.9% | 99.9% | 99.9% |
| DDoS | 99.8% | 99.7% | 99.8% |
| DoS | 99.5% | 99.6% | 99.5% |
| Port Scanning | 98.9% | 98.5% | 98.7% |
| Brute Force | 95.2% | 94.8% | 95.0% |
| Web Attacks | 93.5% | 92.1% | 92.8% |
| Bots | 90.2% | 88.5% | 89.3% |

---

## Limitations

### Known Limitations
1. **Temporal Bias**: Trained on 2017 data; may not generalize to modern attack patterns
2. **Environment Specificity**: Lab-generated attacks may differ from real-world scenarios
3. **Class Imbalance**: Minority classes (Bots, Web Attacks) have lower recall
4. **Feature Dependency**: Requires all 78 input features; missing features degrade performance

### Recommendations
- Regularly retrain with updated attack data
- Use as one component in a defense-in-depth strategy
- Combine with rule-based detection for best results
- Monitor for concept drift in production

---

## Ethical Considerations

### Potential Risks
- **False Positives**: Could flag legitimate traffic as malicious
- **False Negatives**: May miss novel attack variants
- **Privacy**: Operates on network metadata, not packet contents

### Mitigation Strategies
- Human-in-the-loop for critical decisions
- Threshold tuning based on operational requirements
- Regular model evaluation and updates

### Fairness Considerations
- Model treats all network traffic equally
- No demographic or user-identifiable features used
- Classification based solely on traffic characteristics

---

## Technical Specifications

### Input Format
- 52 numerical network flow features (after preprocessing)
- Features include: packet lengths, flow duration, IAT statistics, flags

### Output Format
```json
{
  "prediction": "Normal Traffic",
  "prediction_id": 4,
  "confidence": 0.9999,
  "probabilities": {"class": probability}
}
```

### Inference Performance
- **Latency**: <5ms per prediction
- **Throughput**: 10,000+ predictions/second (batched)
- **Model Size**: ~50MB (JSON format)

---

## Citation

If you use this model in your research, please cite:

```bibtex
@software{watchtower2026,
  author = {Ahmed Tarek},
  title = {WatchTower: Enterprise-Grade Network Intrusion Detection with MLOps},
  year = {2026},
  url = {https://github.com/ahmedtarek-mel/WatchTower}
}
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Jan 2026 | Initial release with 99.877% accuracy |
