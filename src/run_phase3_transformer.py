"""
Phase 3: Transformer Encoder from Scratch.
Implements, trains, and evaluates a pure scratch Transformer Encoder classifier.
Strictly NO nn.MultiheadAttention or pretrained models.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import torch
import torch.nn as nn
from src.data.dataset import get_go_emotions_data
from src.models.transformer import TransformerClassifierFromScratch
from src.training.trainer import fit_model, evaluate_model
from src.evaluation.metrics import plot_training_curves
from src.utils.reproducibility import set_seed, get_device
from src.utils.profiler import count_parameters
from src.utils.mlops_tracker import MLOpsTracker

def run_phase3():
    set_seed(42)
    device = get_device()
    tracker = MLOpsTracker()

    print("\nLoading dataset for Phase 3...")
    data = get_go_emotions_data(batch_size=128)
    vocab = data['vocab']
    label_names = data['label_names']
    num_classes = len(label_names)
    vocab_size = len(vocab)

    print("\nInitializing Transformer Encoder from Scratch...")
    model = TransformerClassifierFromScratch(
        vocab_size=vocab_size,
        embedding_dim=100,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        num_classes=num_classes,
        dropout=0.2,
        max_len=50
    )

    params_info = count_parameters(model)
    tracker.start_run(run_name="Phase3_Transformer_Scratch", phase="Phase3", tags={'model': 'Transformer_Scratch'})
    tracker.log_params({
        'model': 'TransformerClassifierFromScratch',
        'embedding_dim': 100,
        'num_heads': 4,
        'num_layers': 2,
        'd_ff': 256,
        'vocab_size': vocab_size,
        'learning_rate': 5e-4,
        'total_parameters': params_info['total_parameters']
    })

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

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
        model_name="Transformer Encoder (Scratch)"
    )

    # Test evaluation
    test_metrics, y_true, y_pred, _ = evaluate_model(trained_model, data['test_loader'], criterion, device, label_names)
    tracker.log_metrics({
        'test_accuracy': test_metrics['accuracy'],
        'test_macro_f1': test_metrics['macro_f1'],
        'test_loss': test_metrics['loss']
    })

    results = {
        'model': 'Transformer Encoder (Scratch)',
        'parameters': params_info,
        'training_meta': meta,
        'test_metrics': test_metrics
    }

    # Save checkpoint
    torch.save(trained_model.state_dict(), "experiments/checkpoints/phase3_transformer_scratch.pt")

    # Save metrics JSON
    with open('experiments/metrics/phase3_transformer.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved experiments/metrics/phase3_transformer.json")

    # Plot training curves
    plot_training_curves(hist, 'experiments/figures/phase3_transformer_curves.png', 'Transformer Encoder (Scratch)')
    print("Saved experiments/figures/phase3_transformer_curves.png")
    tracker.end_run()

    print("\n=== Phase 3 Summary ===")
    print(f"Total Parameters: {params_info['total_parameters']:,} ({params_info['model_size_mb']:.2f} MB)")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Training Time: {meta['total_training_time_sec']:.1f}s")

if __name__ == '__main__':
    run_phase3()
