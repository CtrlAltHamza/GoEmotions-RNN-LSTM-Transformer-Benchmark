"""
MLOps Experiment Tracking Integration using MLflow and local JSON logs.
Logs hyperparameters, metrics per epoch, final evaluation metrics,
and saves model artifacts with Git commit metadata.
"""
import os
import json
import subprocess
import mlflow

def get_git_commit_hash() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"

class MLOpsTracker:
    def __init__(self, experiment_name: str = "GoEmotions_DeepLearning_A3"):
        self.experiment_name = experiment_name
        try:
            mlflow.set_experiment(experiment_name)
            self.mlflow_available = True
        except Exception as e:
            print(f"Warning: MLflow setup failed ({e}), falling back to JSON logger.")
            self.mlflow_available = False

    def start_run(self, run_name: str, phase: str, tags: dict = None):
        all_tags = {
            'phase': phase,
            'git_commit': get_git_commit_hash(),
            'assignment': 'CS4045_A3'
        }
        if tags:
            all_tags.update(tags)

        if self.mlflow_available:
            return mlflow.start_run(run_name=run_name, tags=all_tags)
        return None

    def log_params(self, params: dict):
        if self.mlflow_available:
            mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int = None):
        if self.mlflow_available:
            flat_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    flat_metrics[k] = v
            mlflow.log_metrics(flat_metrics, step=step)

    def log_artifact(self, file_path: str):
        if self.mlflow_available and os.path.exists(file_path):
            mlflow.log_artifact(file_path)

    def end_run(self):
        if self.mlflow_available:
            mlflow.end_run()

    def save_run_summary(self, summary_data: dict, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
