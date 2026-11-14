"""
Phase 2: Recurrent Baselines (RNN, LSTM, GRU).
Trains Simple RNN, LSTM, and GRU under standardized embedding setup.
Evaluates accuracy, macro-F1, per-class F1, and plots training curves.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import torch
import torch.nn as nn
from src.data.dataset import get_go_emotions_data
from src.models.rnn import SimpleRNNClassifier
from src.models.lstm import LSTMClassifier
from src.models.gru import GRUClassifier
from src.training.trainer import fit_model, evaluate_model
from src.evaluation.metrics import plot_training_curves, plot_confusion_matrix
from src.utils.reproducibility import set_seed, get_device
from src.utils.profiler import count_parameters
from src.utils.mlops_tracker import MLOpsTracker
import matplotlib.pyplot as plt

def run_phase2():
    set_seed(42)
    device = get_device()
    tracker = MLOpsTracker()
    
    print("\nLoading dataset and building train vocabulary...")
    data = get_go_emotions_data(batch_size=128)
    vocab = data['vocab']
    label_names = data['label_names']
    num_classes = len(label_names)
    vocab_size = len(vocab)

    models_dict = {
        'Simple RNN': SimpleRNNClassifier(vocab_size=vocab_size, embedding_dim=100, hidden_dim=128, num_classes=num_classes),
        'LSTM': LSTMClassifier(vocab_size=vocab_size, embedding_dim=100, hidden_dim=128, num_classes=num_classes, readout='last'),
        'GRU': GRUClassifier(vocab_size=vocab_size, embedding_dim=100, hidden_dim=128, num_classes=num_classes)
    }

    results = {}
    histories = {}
    test_predictions = {}
    criterion = nn.CrossEntropyLoss()

    for name, model in models_dict.items():
        print(f"\n==================== Training {name} ====================")
        tracker.start_run(run_name=f"Phase2_{name.replace(' ', '_')}", phase="Phase2", tags={'model': name})
        params_info = count_parameters(model)
        tracker.log_params({
            'model': name,
            'embedding_dim': 100,
            'hidden_dim': 128,
            'vocab_size': vocab_size,
            'learning_rate': 1e-3,
            'total_parameters': params_info['total_parameters'],
            'loss_function': 'StandardCrossEntropy'
        })

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        trained_model, hist, meta = fit_model(
            model=model,
            train_loader=data['train_loader'],
            val_loader=data['val_loader'],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            label_names=label_names,
            epochs=5,
            patience=2,
            model_name=name
        )

        # Final test evaluation
        test_metrics, y_true, y_pred, _ = evaluate_model(trained_model, data['test_loader'], criterion, device, label_names)
        tracker.log_metrics({
            'test_accuracy': test_metrics['accuracy'],
            'test_macro_f1': test_metrics['macro_f1'],
            'test_weighted_f1': test_metrics['weighted_f1'],
            'test_loss': test_metrics['loss']
        })

        results[name] = {
            'parameters': params_info,
            'training_meta': meta,
            'test_metrics': test_metrics
        }
        histories[name] = hist
        test_predictions[name] = {'y_true': y_true, 'y_pred': y_pred}

        # Save checkpoint
        torch.save(trained_model.state_dict(), f"experiments/checkpoints/phase2_{name.lower().replace(' ', '_')}.pt")
        tracker.end_run()

    # Save summary metrics
    with open('experiments/metrics/phase2_recurrent_baselines.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved experiments/metrics/phase2_recurrent_baselines.json")

    # Combined training curves plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    epochs = range(1, len(histories['LSTM']['train_loss']) + 1)
    
    col_idx = 0
    for name in ['Simple RNN', 'LSTM', 'GRU']:
        hist = histories[name]
        ep_range = range(1, len(hist['train_loss']) + 1)
        
        # Loss
        axes[0, col_idx].plot(ep_range, hist['train_loss'], 'b-o', label='Train Loss', linewidth=2)
        axes[0, col_idx].plot(ep_range, hist['val_loss'], 'r--s', label='Val Loss', linewidth=2)
        axes[0, col_idx].set_title(f'{name} - Loss', fontsize=12, fontweight='bold')
        axes[0, col_idx].set_xlabel('Epoch')
        axes[0, col_idx].set_ylabel('Cross-Entropy Loss')
        axes[0, col_idx].grid(True, linestyle=':')
        axes[0, col_idx].legend()

        # Macro-F1
        axes[1, col_idx].plot(ep_range, hist['train_f1'], 'g-o', label='Train Macro-F1', linewidth=2)
        axes[1, col_idx].plot(ep_range, hist['val_f1'], 'm--s', label='Val Macro-F1', linewidth=2)
        axes[1, col_idx].set_title(f'{name} - Macro-F1', fontsize=12, fontweight='bold')
        axes[1, col_idx].set_xlabel('Epoch')
        axes[1, col_idx].set_ylabel('Macro-F1 Score')
        axes[1, col_idx].grid(True, linestyle=':')
        axes[1, col_idx].legend()

        col_idx += 1

    plt.suptitle("Phase 2: Recurrent Baselines Training Curves", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig('experiments/figures/phase2_training_curves.png', dpi=300)
    plt.close()
    print("Saved experiments/figures/phase2_training_curves.png")

    print("\n=== Phase 2 Summary Table ===")
    print(f"{'Model':<15} | {'Params':<10} | {'Test Acc':<10} | {'Test Macro-F1':<15} | {'Training Time':<12}")
    print("-" * 70)
    for name, res in results.items():
        params = res['parameters']['total_parameters']
        acc = res['test_metrics']['accuracy']
        f1 = res['test_metrics']['macro_f1']
        t_time = res['training_meta']['total_training_time_sec']
        print(f"{name:<15} | {params:<10} | {acc:<10.4f} | {f1:<15.4f} | {t_time:<12.1f}s")

if __name__ == '__main__':
    run_phase2()
