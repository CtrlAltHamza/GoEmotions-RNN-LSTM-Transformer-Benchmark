"""
Phase 4: Class Imbalance Solutions and Retraining.
Compares:
1. Standard CrossEntropy (Baseline, leveraged from Phase 2)
2. Class-Weighted CrossEntropy
3. Multi-Class Focal Loss
Evaluates Macro-F1, minority class recall, and confusion matrices.
Selects best strategy and retrains the 4 canonical models for Phases 5 to 8.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import torch
import torch.nn as nn
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.dataset import get_go_emotions_data
from src.models.rnn import SimpleRNNClassifier
from src.models.lstm import LSTMClassifier
from src.models.gru import GRUClassifier
from src.models.transformer import TransformerClassifierFromScratch
from src.training.losses import ClassWeightedCrossEntropyLoss, FocalLoss
from src.training.trainer import fit_model, evaluate_model
from src.evaluation.metrics import plot_confusion_matrix
from src.utils.reproducibility import set_seed, get_device
from src.utils.mlops_tracker import MLOpsTracker

def run_phase4():
    set_seed(42)
    torch.set_num_threads(4)
    device = get_device()
    tracker = MLOpsTracker()

    print("\nLoading dataset for Phase 4...")
    data = get_go_emotions_data(batch_size=128)
    vocab = data['vocab']
    label_names = data['label_names']
    num_classes = len(label_names)
    vocab_size = len(vocab)

    train_labels = [item['label'].item() for item in data['train_ds']]
    label_counts = Counter(train_labels)
    counts_list = [label_counts.get(i, 1) for i in range(num_classes)]

    # Solutions
    loss_strategies = {
        'Baseline (Standard CE)': nn.CrossEntropyLoss(),
        'Class-Weighted CE': ClassWeightedCrossEntropyLoss(counts_list, beta=0.5).to(device),
        'Focal Loss': FocalLoss(gamma=2.0, alpha=None).to(device)
    }

    strategy_results = {}
    strategy_models = {}

    print("\n========== Evaluating Imbalance Solutions on LSTM ==========")

    # 1. Baseline: evaluate already trained Phase 2 LSTM
    baseline_model = LSTMClassifier(vocab_size=vocab_size, embedding_dim=100, hidden_dim=128, num_classes=num_classes, readout='last').to(device)
    baseline_ckpt = "experiments/checkpoints/phase2_lstm.pt"
    if os.path.exists(baseline_ckpt):
        print(f"Loading pre-trained baseline from {baseline_ckpt}...")
        baseline_model.load_state_dict(torch.load(baseline_ckpt, map_location=device))
    else:
        print("Training baseline LSTM...")
        opt = torch.optim.Adam(baseline_model.parameters(), lr=1e-3, weight_decay=1e-5)
        baseline_model, _, _ = fit_model(baseline_model, data['train_loader'], data['val_loader'], loss_strategies['Baseline (Standard CE)'], opt, device, label_names, epochs=5, patience=2, model_name="LSTM Baseline")

    test_metrics, y_true, y_pred, _ = evaluate_model(baseline_model, data['test_loader'], loss_strategies['Baseline (Standard CE)'], device, label_names)
    strategy_results['Baseline (Standard CE)'] = test_metrics
    strategy_models['Baseline (Standard CE)'] = baseline_model
    plot_confusion_matrix(y_true, y_pred, label_names, "experiments/figures/phase4_cm_baseline.png", title="Confusion Matrix: Baseline (Standard CE)")

    # 2. Train Class-Weighted CE and Focal Loss
    for strat_name in ['Class-Weighted CE', 'Focal Loss']:
        print(f"\n--- Training Strategy: {strat_name} ---")
        tracker.start_run(run_name=f"Phase4_{strat_name.replace(' ', '_')}", phase="Phase4", tags={'strategy': strat_name})
        criterion = loss_strategies[strat_name]
        
        set_seed(42)
        model = LSTMClassifier(vocab_size=vocab_size, embedding_dim=100, hidden_dim=128, num_classes=num_classes, readout='last').to(device)
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
            model_name=f"LSTM ({strat_name})"
        )

        test_m, y_t, y_p, _ = evaluate_model(trained_model, data['test_loader'], criterion, device, label_names)
        tracker.log_metrics({
            'strategy': strat_name,
            'test_accuracy': test_m['accuracy'],
            'test_macro_f1': test_m['macro_f1'],
            'test_macro_recall': test_m['macro_recall']
        })

        strategy_results[strat_name] = test_m
        strategy_models[strat_name] = trained_model

        cm_name = "experiments/figures/phase4_cm_class-weighted_ce.png" if "Weighted" in strat_name else "experiments/figures/phase4_cm_focal_loss.png"
        plot_confusion_matrix(y_t, y_p, label_names, cm_name, title=f"Confusion Matrix: {strat_name}")
        tracker.end_run()

    # Determine best strategy
    best_strategy_name = max(strategy_results.keys(), key=lambda k: strategy_results[k]['macro_f1'])
    best_criterion = loss_strategies[best_strategy_name]
    print(f"\n>>> Selected Best Imbalance Strategy: '{best_strategy_name}' (Macro-F1: {strategy_results[best_strategy_name]['macro_f1']:.4f})")

    # Plot Minority Classes Recall Comparison
    minority_classes = ['grief', 'pride', 'nervousness', 'relief', 'embarrassment']
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(minority_classes))
    width = 0.25

    for idx, (s_name, s_res) in enumerate(strategy_results.items()):
        recalls = [s_res['minority_classes_recall'].get(c, 0.0) * 100 for c in minority_classes]
        offset = (idx - 1) * width
        ax.bar([p + offset for p in x], recalls, width=width, label=s_name)

    ax.set_xticks(x)
    ax.set_xticklabels(minority_classes, fontsize=11, fontweight='bold')
    ax.set_title('Minority Emotion Classes: Recall Comparison Across Strategies', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Minority Emotion Class', fontsize=11, fontweight='bold')
    ax.set_ylabel('Recall (%)', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle=':', axis='y')
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig('experiments/figures/phase4_minority_recall_comparison.png', dpi=300)
    plt.close()
    print("Saved experiments/figures/phase4_minority_recall_comparison.png")

    # Save canonical LSTM directly from best model
    os.makedirs('experiments/checkpoints', exist_ok=True)
    best_lstm_model = strategy_models[best_strategy_name]
    torch.save(best_lstm_model.state_dict(), "experiments/checkpoints/canonical_lstm.pt")
    print("Saved canonical LSTM to experiments/checkpoints/canonical_lstm.pt")

    # Retrain remaining canonical models: Simple RNN, GRU, and Transformer
    canonical_models_to_train = {
        'Simple RNN': SimpleRNNClassifier(vocab_size=vocab_size, embedding_dim=100, hidden_dim=128, num_classes=num_classes),
        'GRU': GRUClassifier(vocab_size=vocab_size, embedding_dim=100, hidden_dim=128, num_classes=num_classes),
        'Transformer': TransformerClassifierFromScratch(vocab_size=vocab_size, embedding_dim=100, num_heads=4, num_layers=2, d_ff=256, num_classes=num_classes)
    }

    retrained_results = {'LSTM': strategy_results[best_strategy_name]}

    for name, model in canonical_models_to_train.items():
        print(f"\n--- Retraining Canonical {name} with '{best_strategy_name}' ---")
        set_seed(42)
        lr = 5e-4 if name == 'Transformer' else 1e-3
        opt_class = torch.optim.AdamW if name == 'Transformer' else torch.optim.Adam
        optimizer = opt_class(model.parameters(), lr=lr, weight_decay=1e-5)

        trained_model, hist, meta = fit_model(
            model=model,
            train_loader=data['train_loader'],
            val_loader=data['val_loader'],
            criterion=best_criterion,
            optimizer=optimizer,
            device=device,
            label_names=label_names,
            epochs=4,
            patience=2,
            model_name=f"Canonical {name}"
        )

        test_metrics, _, _, _ = evaluate_model(trained_model, data['test_loader'], best_criterion, device, label_names)
        retrained_results[name] = test_metrics

        ckpt_path = f"experiments/checkpoints/canonical_{name.lower().replace(' ', '_')}.pt"
        torch.save(trained_model.state_dict(), ckpt_path)
        print(f"Saved canonical checkpoint to {ckpt_path}")

    # Save summary metrics
    summary = {
        'loss_strategies_compared': {k: {'accuracy': v['accuracy'], 'macro_f1': v['macro_f1'], 'minority_recalls': v['minority_classes_recall']} for k, v in strategy_results.items()},
        'selected_best_strategy': best_strategy_name,
        'retrained_canonical_models': retrained_results
    }
    with open('experiments/metrics/phase4_imbalance_comparison.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nSaved experiments/metrics/phase4_imbalance_comparison.json")

    print("\n=== Phase 4 Summary Table ===")
    print(f"{'Strategy':<25} | {'Accuracy':<10} | {'Macro-F1':<10} | {'Macro-Recall':<12}")
    print("-" * 65)
    for s_name, res in strategy_results.items():
        print(f"{s_name:<25} | {res['accuracy']:<10.4f} | {res['macro_f1']:<10.4f} | {res['macro_recall']:<12.4f}")

if __name__ == '__main__':
    run_phase4()
