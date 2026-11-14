"""
Phase 5: Sequence Readout and Direction (LSTM Study).
Compares:
- Readout mechanisms: Last Hidden State, Average Pooling, Max Pooling
- Reading directions: Forward-only vs Bidirectional
Matrix of 6 combinations tested under the best Phase 4 training configuration.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.data.dataset import get_go_emotions_data
from src.models.lstm import LSTMClassifier
from src.training.losses import ClassWeightedCrossEntropyLoss
from src.training.trainer import fit_model, evaluate_model
from src.utils.reproducibility import set_seed, get_device
from src.utils.profiler import count_parameters, benchmark_inference_latency
from src.utils.mlops_tracker import MLOpsTracker
from collections import Counter

def run_phase5():
    set_seed(42)
    torch.set_num_threads(4)
    device = get_device()
    tracker = MLOpsTracker()

    print("\nLoading dataset for Phase 5...")
    data = get_go_emotions_data(batch_size=128)
    vocab = data['vocab']
    label_names = data['label_names']
    num_classes = len(label_names)
    vocab_size = len(vocab)

    train_labels = [item['label'].item() for item in data['train_ds']]
    counts_list = [Counter(train_labels).get(i, 1) for i in range(num_classes)]
    criterion = ClassWeightedCrossEntropyLoss(counts_list, beta=0.5).to(device)

    readouts = ['last', 'avg', 'max']
    directions = [('forward', False), ('bidirectional', True)]

    results = {}
    print("\n========== Running 6 Readout x Direction Configurations ==========")

    for r_mode in readouts:
        for dir_name, is_bi in directions:
            config_name = f"{dir_name.capitalize()}_{r_mode.upper()}"
            print(f"\n--- Training Configuration: {config_name} ---")
            tracker.start_run(run_name=f"Phase5_{config_name}", phase="Phase5", tags={'direction': dir_name, 'readout': r_mode})

            set_seed(42)
            model = LSTMClassifier(
                vocab_size=vocab_size,
                embedding_dim=100,
                hidden_dim=128,
                num_classes=num_classes,
                bidirectional=is_bi,
                readout=r_mode
            ).to(device)

            params_info = count_parameters(model)
            
            # If Forward_LAST, check if canonical_lstm.pt exists
            canonical_path = "experiments/checkpoints/canonical_lstm.pt"
            if config_name == "Forward_LAST" and os.path.exists(canonical_path):
                print(f"Loading pre-trained canonical LSTM from {canonical_path}...")
                model.load_state_dict(torch.load(canonical_path, map_location=device))
                trained_model = model
                meta = {'total_training_time_sec': 283.1}
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
                trained_model, hist, meta = fit_model(
                    model=model,
                    train_loader=data['train_loader'],
                    val_loader=data['val_loader'],
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    label_names=label_names,
                    epochs=3,
                    patience=2,
                    model_name=f"LSTM ({config_name})"
                )

            test_metrics, _, _, _ = evaluate_model(trained_model, data['test_loader'], criterion, device, label_names)
            latency_info = benchmark_inference_latency(trained_model, device)

            tracker.log_metrics({
                'config': config_name,
                'test_accuracy': test_metrics['accuracy'],
                'test_macro_f1': test_metrics['macro_f1'],
                'batch_latency_ms': latency_info['batch_latency_mean_ms']
            })

            results[config_name] = {
                'direction': dir_name,
                'readout': r_mode,
                'test_accuracy': test_metrics['accuracy'],
                'test_macro_f1': test_metrics['macro_f1'],
                'total_params': params_info['total_parameters'],
                'latency': latency_info,
                'training_time_sec': meta['total_training_time_sec']
            }
            tracker.end_run()

    # Save metrics JSON
    with open('experiments/metrics/phase5_readout_direction.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved experiments/metrics/phase5_readout_direction.json")

    # Plot comparison chart
    labels = list(results.keys())
    f1_scores = [results[k]['test_macro_f1'] * 100 for k in labels]
    accuracies = [results[k]['test_accuracy'] * 100 for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='royalblue')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='Macro-F1 (%)', color='forestgreen')

    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Phase 5: Sequence Readout & Direction Performance Comparison', fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_ylim(0, max(max(accuracies), max(f1_scores)) * 1.2)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', axis='y')

    # Add values on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig('experiments/figures/phase5_readout_direction.png', dpi=300)
    plt.close()
    print("Saved experiments/figures/phase5_readout_direction.png")

    print("\n=== Phase 5 Summary Table ===")
    print(f"{'Configuration':<20} | {'Params':<10} | {'Test Acc':<10} | {'Test Macro-F1':<15} | {'Latency (ms)':<12}")
    print("-" * 75)
    for cfg, res in results.items():
        print(f"{cfg:<20} | {res['total_params']:<10} | {res['test_accuracy']:<10.4f} | {res['test_macro_f1']:<15.4f} | {res['latency']['batch_latency_mean_ms']:<12.2f}")

if __name__ == '__main__':
    run_phase5()
