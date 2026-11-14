"""
Phase 8: Comprehensive Model Benchmarking, Statistical Significance, and Noise Robustness.
Compares the 4 canonical models on Accuracy, Macro-F1, Size, Latency, and Training Cost.
Applies McNemar's Statistical Significance Test between Best Recurrent Model and Scratch Transformer.
Executes Test-Time Noise Robustness Experiment across noise levels [0%, 5%, 10%, 20%, 30%].
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.data.dataset import get_go_emotions_data
from src.models.rnn import SimpleRNNClassifier
from src.models.lstm import LSTMClassifier
from src.models.gru import GRUClassifier
from src.models.transformer import TransformerClassifierFromScratch
from src.evaluation.metrics import compute_all_metrics, run_mcnemar_test
from src.evaluation.robustness import evaluate_on_noisy_data, plot_noise_robustness
from src.utils.profiler import count_parameters, benchmark_inference_latency
from src.utils.reproducibility import set_seed, get_device
from src.utils.mlops_tracker import MLOpsTracker

def run_phase8():
    set_seed(42)
    device = get_device()
    tracker = MLOpsTracker()
    tracker.start_run(run_name="Phase8_Benchmark_Robustness", phase="Phase8")

    print("\nLoading dataset for Phase 8...")
    data = get_go_emotions_data(batch_size=128)
    vocab = data['vocab']
    label_names = data['label_names']
    num_classes = len(label_names)
    vocab_size = len(vocab)
    test_loader = data['test_loader']
    test_ds = data['test_ds']

    models = {
        'Simple RNN': SimpleRNNClassifier(vocab_size=vocab_size, embedding_dim=100, hidden_dim=128, num_classes=num_classes).to(device),
        'LSTM': LSTMClassifier(vocab_size=vocab_size, embedding_dim=100, hidden_dim=128, num_classes=num_classes, readout='last').to(device),
        'GRU': GRUClassifier(vocab_size=vocab_size, embedding_dim=100, hidden_dim=128, num_classes=num_classes).to(device),
        'Transformer': TransformerClassifierFromScratch(vocab_size=vocab_size, embedding_dim=100, num_heads=4, num_layers=2, d_ff=256, num_classes=num_classes).to(device)
    }

    # Load canonical checkpoints
    for name, model in models.items():
        ckpt_path = f"experiments/checkpoints/canonical_{name.lower().replace(' ', '_')}.pt"
        if not os.path.exists(ckpt_path):
            ckpt_path = f"experiments/checkpoints/phase2_{name.lower().replace(' ', '_')}.pt" if name != 'Transformer' else "experiments/checkpoints/phase3_transformer_scratch.pt"
        print(f"Loading {name} weights from {ckpt_path}...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

    # Load Phase 4 metadata for training time
    phase4_meta = {}
    if os.path.exists('experiments/metrics/phase4_imbalance_comparison.json'):
        with open('experiments/metrics/phase4_imbalance_comparison.json') as f:
            p4_data = json.load(f)
            phase4_meta = p4_data.get('retrained_canonical_models', {})

    benchmark_table = {}
    model_predictions = {}
    ground_truth = None

    print("\n========== Model Benchmarking (Accuracy, F1, Size, Latency) ==========")
    for name, model in models.items():
        params_info = count_parameters(model)
        latency_info = benchmark_inference_latency(model, device)

        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in test_loader:
                inp = batch['input_ids'].to(device)
                lengths = batch['length'].to(device)
                lbl = batch['label'].to(device)
                logits = model(inp, lengths)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(lbl.cpu().numpy())

        if ground_truth is None:
            ground_truth = np.array(all_targets)
        model_predictions[name] = np.array(all_preds)

        metrics = compute_all_metrics(all_targets, all_preds, label_names)
        t_time = phase4_meta.get(name, {}).get('training_time_sec', 0.0)

        benchmark_table[name] = {
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1'],
            'total_parameters': params_info['total_parameters'],
            'model_size_mb': params_info['model_size_mb'],
            'batch_latency_ms': latency_info['batch_latency_mean_ms'],
            'per_sample_latency_ms': latency_info['per_sample_latency_ms'],
            'training_time_sec': t_time
        }

    # Print Comparison Table
    print("\n" + "=" * 90)
    print(f"{'Model':<15} | {'Acc':<8} | {'Macro-F1':<10} | {'Params':<10} | {'Size (MB)':<10} | {'Latency (ms)':<12} | {'Train Time':<10}")
    print("-" * 90)
    for name, bm in benchmark_table.items():
        print(f"{name:<15} | {bm['accuracy']:<8.4f} | {bm['macro_f1']:<10.4f} | {bm['total_parameters']:<10,d} | {bm['model_size_mb']:<10.2f} | {bm['batch_latency_ms']:<12.2f} | {bm['training_time_sec']:<10.1f}s")
    print("=" * 90)

    # Statistical Significance Test: Best Recurrent Model vs Scratch Transformer
    recurrent_candidates = ['Simple RNN', 'LSTM', 'GRU']
    best_recurrent_name = max(recurrent_candidates, key=lambda m: benchmark_table[m]['macro_f1'])
    print(f"\nRunning McNemar's Test between Best Recurrent Model ('{best_recurrent_name}') and 'Transformer'...")

    mcnemar_result = run_mcnemar_test(
        y_true=ground_truth,
        y_pred_a=model_predictions[best_recurrent_name],
        y_pred_b=model_predictions['Transformer']
    )
    print(f"McNemar's Test Results: chi2 = {mcnemar_result['chi2_statistic']}, p-value = {mcnemar_result['p_value_formatted']}")
    print(f"Statistically significant at alpha=0.05: {mcnemar_result['is_statistically_significant_at_05']}")

    # Test-Time Noise Experiment
    print("\n========== Running Test-Time Noise Robustness Experiment ==========")
    noise_rates = [0.0, 0.05, 0.1, 0.2, 0.3]
    robustness_data = {'noise_rates': noise_rates, 'models': {}}

    for name, model in models.items():
        print(f"\n--- Testing Robustness for {name} ---")
        rob_res = evaluate_on_noisy_data(
            model=model,
            test_ds=test_ds,
            vocab=vocab,
            label_names=label_names,
            device=device,
            noise_rates=noise_rates
        )
        robustness_data['models'][name] = rob_res

    plot_noise_robustness(robustness_data, 'experiments/figures/phase8_noise_robustness.png')
    print("Saved experiments/figures/phase8_noise_robustness.png")

    # Save summary metrics
    summary = {
        'benchmark_comparison_table': benchmark_table,
        'best_recurrent_model': best_recurrent_name,
        'mcnemar_statistical_test': mcnemar_result,
        'noise_robustness_experiment': robustness_data
    }

    with open('experiments/metrics/phase8_benchmark_robustness.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("Saved experiments/metrics/phase8_benchmark_robustness.json")
    tracker.end_run()

if __name__ == '__main__':
    run_phase8()
