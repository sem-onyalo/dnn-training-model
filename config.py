import os

cloud_storage_container = os.environ.get("CLOUD_STORAGE_CONTAINER", "gan-training-storage")
cloud_storage_account = os.environ.get("CLOUD_STORAGE_ACCOUNT", "5d1796b484c2431eb911")
epoch_directory = os.environ.get("EPOCH_DIRECTORY", "epoch")
generated_samples_file = os.environ.get("GENERATED_SAMPLES_FILE", "generated.png")
hyperparameters_file = os.environ.get("HYPERPARAMETERS_FILE", "hyperparameters.json")
loss_accuracy_file = os.environ.get("LOSS_ACCURACY_FILE", "loss_accuracy.png")
metrics_file = os.environ.get("METRICS_FILE", "metrics.csv")
model_file = os.environ.get("MODEL_FILE", "model.h5")
summary_file = os.environ.get("SUMMARY_FILE", "summary.json")
target_samples_file = os.environ.get("TARGET_SAMPLES_FILE", "target.png")
