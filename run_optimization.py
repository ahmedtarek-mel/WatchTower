"""Run Optuna hyperparameter optimization to beat 99.89% benchmark."""

from src.data.ingestion import load_dataset
from src.data.preprocessing import DataPreprocessor, create_train_val_test_split
from src.training.hyperopt import OptunaOptimizer
from src.training.train import XGBoostTrainer

print("=" * 60)
print("OPTUNA HYPERPARAMETER OPTIMIZATION")
print("Target: Beat 99.89% accuracy benchmark")
print("=" * 60)

# Load 10% sample for faster optimization while maintaining representative data
df = load_dataset(sample_frac=0.10)

# Preprocess
preprocessor = DataPreprocessor()
X, y = preprocessor.fit_transform(df)

# Split
X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(X, y)

# Run Optuna optimization with 30 trials
print("\nStarting 30 Optuna trials...\n")
optimizer = OptunaOptimizer(n_trials=30)
best_params = optimizer.optimize(X_train, y_train, X_val, y_val)

# Train final model with best params
print("\n" + "=" * 60)
print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
print("=" * 60)

trainer = XGBoostTrainer(
    n_estimators=best_params.get("n_estimators", 100),
    max_depth=best_params.get("max_depth", 6),
    learning_rate=best_params.get("learning_rate", 0.1),
)
model = trainer.train(X_train, y_train, X_val, y_val)

# Final evaluation
print("\n" + "=" * 60)
print("FINAL EVALUATION ON TEST SET")
print("=" * 60)
metrics = trainer.evaluate(X_test, y_test, "Test")

accuracy = metrics["accuracy"]
print("\n" + "=" * 60)
print(f"RESULT: {accuracy*100:.4f}% accuracy")
print("Benchmark: 99.89%")
if accuracy >= 0.9989:
    print("STATUS: BENCHMARK BEATEN!")
else:
    gap = (0.9989 - accuracy) * 100
    print(f"Gap: {gap:.4f}%")
print("=" * 60)

# Save best model
trainer.save_model()
print("\nModel saved!")
