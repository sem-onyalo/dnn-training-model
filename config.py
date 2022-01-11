import os

cloud_blob_root = os.environ.get("CLOUD_BLOB_ROOT", "gan-training-storage")
epoch_directory = os.environ.get("EPOCH_DIRECTORY", "epoch")
generated_samples_file = os.environ.get("GENERATED_SAMPLES_FILE", "generated.png")
hyperparameters_file = os.environ.get("HYPERPARAMETERS_FILE", "hyperparameters.json")
loss_accuracy_file = os.environ.get("LOSS_ACCURACY_FILE", "loss_accuracy.png")
metrics_file = os.environ.get("METRICS_FILE", "metrics.csv")
model_file = os.environ.get("MODEL_FILE", "model.h5")
summary_file = os.environ.get("SUMMARY_FILE", "summary.json")
target_samples_file = os.environ.get("TARGET_SAMPLES_FILE", "target.png")
