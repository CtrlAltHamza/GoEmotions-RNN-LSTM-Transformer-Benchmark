"""
Training loop, validation tracking, early stopping, and evaluation runner.
Ensures validation split is used for all tuning and model checkpointing.
"""
import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from src.evaluation.metrics import compute_all_metrics

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        lengths = batch['length'].to(device)
        targets = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, lengths)
        loss = criterion(logits, targets)
        loss.backward()

        # Gradient clipping to stabilize recurrent networks
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    epoch_acc = accuracy_score(all_targets, all_preds)

    return epoch_loss, epoch_acc, epoch_f1

def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, label_names: list[str]):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    start_time = time.time()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            lengths = batch['length'].to(device)
            targets = batch['label'].to(device)

            logits = model(input_ids, lengths)
            loss = criterion(logits, targets)

            total_loss += loss.item() * input_ids.size(0)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np_argmax = torch.argmax(logits, dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs)

    eval_time = time.time() - start_time
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_all_metrics(all_targets, all_preds, label_names)
    metrics['loss'] = round(avg_loss, 4)
    metrics['eval_time_sec'] = round(eval_time, 3)
    metrics['ms_per_sample'] = round((eval_time / len(loader.dataset)) * 1000, 3)

    return metrics, all_targets, all_preds, all_probs

def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_names: list[str],
    epochs: int = 10,
    patience: int = 3,
    model_name: str = "Model"
):
    print(f"\n--- Starting Training: {model_name} (Max Epochs: {epochs}, Patience: {patience}) ---")
    model.to(device)

    best_val_f1 = -1.0
    best_weights = None
    best_epoch = 0
    patience_counter = 0

    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'epoch_times': []
    }

    total_start = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _, _, _ = evaluate_model(model, val_loader, criterion, device, label_names)
        ep_time = time.time() - ep_start

        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        val_f1 = val_metrics['macro_f1']

        history['train_loss'].append(round(train_loss, 4))
        history['train_acc'].append(round(train_acc, 4))
        history['train_f1'].append(round(train_f1, 4))
        history['val_loss'].append(round(val_loss, 4))
        history['val_acc'].append(round(val_acc, 4))
        history['val_f1'].append(round(val_f1, 4))
        history['epoch_times'].append(round(ep_time, 2))

        print(f"Epoch {epoch:02d}/{epochs:02d} [{ep_time:.1f}s] - Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # Check early stopping condition based strictly on validation split
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  --> [Checkpointed] Best Validation Macro-F1: {best_val_f1:.4f} at epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  --> Early stopping triggered at epoch {epoch} (no val improvement for {patience} epochs).")
                break

    total_train_time = time.time() - total_start
    print(f"Training completed in {total_train_time:.1f}s. Best Epoch: {best_epoch} with Val Macro-F1: {best_val_f1:.4f}")

    # Load best checkpoint
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history, {
        'best_epoch': best_epoch,
        'best_val_macro_f1': best_val_f1,
        'total_training_time_sec': round(total_train_time, 2),
        'avg_epoch_time_sec': round(float(sum(history['epoch_times']) / len(history['epoch_times'])), 2)
    }
