"""
Phase 6: Word Representation: Random vs Pretrained Embeddings (GloVe 100d).
Evaluates the 4 canonical models under:
1. Random Embeddings (Trained from scratch)
2. Pretrained GloVe (Frozen)
3. Pretrained GloVe (Fine-tuned)
Reports vocabulary coverage and class-level performance gains.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from src.data.dataset import get_go_emotions_data
from src.models.rnn import SimpleRNNClassifier
from src.models.lstm import LSTMClassifier
from src.models.gru import GRUClassifier
from src.models.transformer import TransformerClassifierFromScratch
from src.models.embeddings import build_glove_matrix
from src.training.losses import ClassWeightedCrossEntropyLoss
from src.training.trainer import fit_model, evaluate_model
from src.utils.reproducibility import set_seed, get_device
from src.utils.mlops_tracker import MLOpsTracker
from collections import Counter

def apply_pretrained_weights(model: nn.Module, weights_matrix: torch.Tensor, freeze: bool):
    with torch.no_grad():
        model.embedding.weight.copy_(weights_matrix)
    model.embedding.weight.requires_grad = not freeze

def run_phase6():
    set_seed(42)
    device = get_device()
    tracker = MLOpsTracker()

    print("\nLoading dataset for Phase 6...")
    data = get_go_emotions_data(batch_size=128)
    vocab = data['vocab']
    label_names = data['label_names']
    num_classes = len(label_names)
    vocab_size = len(vocab)

    train_labels = [item['label'].item() for item in data['train_ds']]
    counts_list = [Counter(train_labels).get(i, 1) for i in range(num_classes)]
    criterion = ClassWeightedCrossEntropyLoss(counts_list, beta=0.5).to(device)

    # Build GloVe weights matrix
    glove_weights, coverage_info = build_glove_matrix(vocab, embedding_dim=100)

    model_factories = {
        'Simple RNN': lambda: SimpleRNNClassifier(vocab_size=vocab_size, embedding_dim=100, hidden_dim=128, num_classes=num_classes),
        'LSTM': lambda: LSTMClassifier(vocab_size=vocab_size, embedding_dim=100, hidden_dim=128, num_classes=num_classes, readout='last'),
        'GRU': lambda: GRUClassifier(vocab_size=vocab_size, embedding_dim=100, hidden_dim=128, num_classes=num_classes),
        'Transformer': lambda: TransformerClassifierFromScratch(vocab_size=vocab_size, embedding_dim=100, num_heads=4, num_layers=2, d_ff=256, num_classes=num_classes)
    }

    results = {
        'coverage_info': coverage_info,
        'experiments': {}
    }

    modes = ['Random', 'Pretrained_Frozen', 'Pretrained_Finetuned']

    for model_name, factory in model_factories.items():
        results['experiments'][model_name] = {}
        for mode in modes:
            exp_name = f"{model_name}_{mode}"
            print(f"\n==================== Running {exp_name} ====================")
            tracker.start_run(run_name=f"Phase6_{exp_name}", phase="Phase6", tags={'model': model_name, 'embedding_mode': mode})

            set_seed(42)
            model = factory()

            if mode == 'Pretrained_Frozen':
                apply_pretrained_weights(model, glove_weights, freeze=True)
            elif mode == 'Pretrained_Finetuned':
                apply_pretrained_weights(model, glove_weights, freeze=False)
            # If Random, weights remain initialized from scratch

            lr = 5e-4 if model_name == 'Transformer' else 1e-3
            opt_class = torch.optim.AdamW if model_name == 'Transformer' else torch.optim.Adam
            # If frozen, only optimize parameters that require grad
            optimizer = opt_class(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)

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
                model_name=exp_name
            )

            test_metrics, _, _, _ = evaluate_model(trained_model, data['test_loader'], criterion, device, label_names)
            tracker.log_metrics({
                'test_accuracy': test_metrics['accuracy'],
                'test_macro_f1': test_metrics['macro_f1']
            })

            results['experiments'][model_name][mode] = {
                'accuracy': test_metrics['accuracy'],
                'macro_f1': test_metrics['macro_f1'],
                'minority_recall': test_metrics['minority_classes_recall'],
                'training_time_sec': meta['total_training_time_sec']
            }
            tracker.end_run()

    # Save metrics JSON
    with open('experiments/metrics/phase6_embeddings.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved experiments/metrics/phase6_embeddings.json")

    # Plot comparison chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    model_names = list(model_factories.keys())
    x = np.arange(len(model_names))
    width = 0.25

    for idx, mode in enumerate(modes):
        accs = [results['experiments'][m][mode]['accuracy'] * 100 for m in model_names]
        f1s = [results['experiments'][m][mode]['macro_f1'] * 100 for m in model_names]
        offset = (idx - 1) * width

        label_fmt = mode.replace('_', ' ')
        ax1.bar(x + offset, accs, width=width, label=label_fmt)
        ax2.bar(x + offset, f1s, width=width, label=label_fmt)

    ax1.set_title('Test Accuracy: Random vs Pretrained GloVe', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Model Architecture', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=10, fontweight='bold')
    ax1.grid(True, linestyle=':', axis='y')
    ax1.legend()

    ax2.set_title('Test Macro-F1: Random vs Pretrained GloVe', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Model Architecture', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Macro-F1 (%)', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle=':', axis='y')
    ax2.legend()

    plt.tight_layout()
    fig.savefig('experiments/figures/phase6_embedding_comparison.png', dpi=300)
    plt.close()
    print("Saved experiments/figures/phase6_embedding_comparison.png")

if __name__ == '__main__':
    run_phase6()
