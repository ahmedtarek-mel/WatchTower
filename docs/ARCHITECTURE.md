# WatchTower Architecture

This document provides detailed architecture diagrams for the WatchTower MLOps pipeline.

## System Overview

```mermaid
flowchart TB
    subgraph Data["Data Layer"]
        CSV[(CICIDS2017<br/>Dataset)]
        ING[Data Ingestion]
        VAL[Validation]
        PRE[Preprocessing]
    end

    subgraph Training["Training Pipeline"]
        OPT[Optuna<br/>Optimization]
        XGB[XGBoost<br/>GPU Training]
        EVAL[Evaluation]
        MLF[(MLflow<br/>Tracking)]
    end

    subgraph Orchestration["Orchestration"]
        PRF[Prefect<br/>Workflows]
    end

    subgraph Serving["Serving Layer"]
        API[FastAPI<br/>API]
        MOD[(Model<br/>Registry)]
        PROM[Prometheus<br/>Metrics]
    end

    subgraph Clients["Clients"]
        WEB[Web Apps]
        CLI[CLI Tools]
        INT[Integrations]
    end

    CSV --> ING --> VAL --> PRE
    PRE --> OPT
    OPT <--> XGB
    XGB --> EVAL
    XGB --> MLF
    EVAL --> MOD

    PRF -.-> ING
    PRF -.-> XGB
    PRF -.-> EVAL

    MOD --> API
    API --> PROM
    API --> WEB
    API --> CLI
    API --> INT
```

## Data Flow

```mermaid
flowchart LR
    subgraph Input
        RAW[Raw CSV<br/>2.8M rows]
    end

    subgraph Processing
        LOAD[Load with<br/>Progress]
        CLEAN[Clean<br/>Inf/NaN]
        ENCODE[Label<br/>Encode]
        SCALE[Standard<br/>Scale]
        SPLIT[Train/Val/Test<br/>Split]
    end

    subgraph Output
        TRAIN[Training Set<br/>70%]
        VAL[Validation Set<br/>15%]
        TEST[Test Set<br/>15%]
    end

    RAW --> LOAD --> CLEAN --> ENCODE --> SCALE --> SPLIT
    SPLIT --> TRAIN
    SPLIT --> VAL
    SPLIT --> TEST
```

## Training Pipeline

```mermaid
flowchart TB
    subgraph Optimization
        STUDY[Create Optuna<br/>Study]
        TRIAL[Sample<br/>Hyperparameters]
        FIT[Train XGBoost]
        SCORE[Evaluate<br/>Accuracy]
        PRUNE{Early<br/>Stopping?}
    end

    subgraph BestModel
        BEST[Best<br/>Hyperparameters]
        FINAL[Final<br/>Training]
        SAVE[Save Model<br/>JSON]
    end

    STUDY --> TRIAL --> FIT --> SCORE --> PRUNE
    PRUNE -->|No| TRIAL
    PRUNE -->|30 trials| BEST
    BEST --> FINAL --> SAVE
```

## API Request Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant A as FastAPI
    participant M as Model
    participant P as Prometheus

    C->>A: POST /predict
    A->>A: Validate Request
    A->>M: Load Features
    M->>M: Predict
    M-->>A: Probabilities
    A->>P: Log Metrics
    A-->>C: JSON Response
```

## Deployment Architecture

```mermaid
flowchart TB
    subgraph Cloud["Cloud/On-Prem"]
        subgraph Docker["Docker Compose"]
            API[FastAPI<br/>:8000]
            MLF[MLflow<br/>:5000]
            PRF[Prefect<br/>:4200]
            PROM[Prometheus<br/>:9090]
            GRAF[Grafana<br/>:3000]
        end
        
        subgraph Storage
            DB[(SQLite/<br/>PostgreSQL)]
            VOL[(Model<br/>Volume)]
        end
    end

    subgraph External
        GH[GitHub<br/>Actions]
        REG[Container<br/>Registry]
    end

    GH -->|CI/CD| REG -->|Deploy| Docker
    API --> VOL
    MLF --> DB
    PRF --> DB
    PROM --> API
    GRAF --> PROM
```

## Class Hierarchy

```mermaid
classDiagram
    class Settings {
        +project_root: Path
        +data_dir: Path
        +models_dir: Path
        +gpu_available: bool
        +print_config()
    }

    class DataPreprocessor {
        +scaler: StandardScaler
        +label_encoder: LabelEncoder
        +fit_transform(df)
        +transform(df)
        +decode_labels(y)
    }

    class XGBoostTrainer {
        +model: XGBClassifier
        +device: str
        +train(X, y, X_val, y_val)
        +predict(X)
        +evaluate(X, y)
        +save_model()
    }

    class OptunaOptimizer {
        +study: Study
        +n_trials: int
        +optimize(X, y, X_val, y_val)
        +get_best_trainer()
    }

    class ModelInference {
        +model: XGBClassifier
        +load_model()
        +predict(features)
        +predict_batch(samples)
    }

    Settings --> DataPreprocessor
    Settings --> XGBoostTrainer
    OptunaOptimizer --> XGBoostTrainer
    XGBoostTrainer --> ModelInference
```

---

## Quick Reference

| Component | Technology | Port |
|-----------|------------|------|
| API | FastAPI + Uvicorn | 8000 |
| Experiments | MLflow | 5000 |
| Workflows | Prefect | 4200 |
| Metrics | Prometheus | 9090 |
| Dashboards | Grafana | 3000 |

---

## Author

**Ahmed Tarek** - Data Scientist & Machine Learning Engineer

- [LinkedIn](https://www.linkedin.com/in/ahmed-tarek-mel)
- [Portfolio](https://www.datascienceportfol.io/AhmedTarek)
- [Email](mailto:ahmedtarekmel@gmail.com)
