"""
Comprehensive evaluation metrics, confusion matrix, statistical significance testing,
and visualization utilities for GoEmotions classification.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from scipy.stats import chi2

def compute_all_metrics(y_true: list | np.ndarray, y_pred: list | np.ndarray, label_names: list[str]) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    num_classes = len(label_names)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0).tolist()
    per_class_recall = recall_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0).tolist()
    per_class_precision = precision_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0).tolist()

    class_metrics = {}
    for idx, name in enumerate(label_names):
        class_metrics[name] = {
            'f1': round(float(per_class_f1[idx]), 4),
            'recall': round(float(per_class_recall[idx]), 4),
            'precision': round(float(per_class_precision[idx]), 4)
        }

    # Small/minority classes recall summary (bottom 5 classes)
    minority_classes = ['grief', 'pride', 'nervousness', 'relief', 'embarrassment']
    minority_recalls = {c: class_metrics.get(c, {}).get('recall', 0.0) for c in minority_classes}

    return {
        'accuracy': round(acc, 4),
        'macro_f1': round(macro_f1, 4),
        'weighted_f1': round(weighted_f1, 4),
        'macro_recall': round(macro_recall, 4),
        'per_class_metrics': class_metrics,
        'minority_classes_recall': minority_recalls
    }

def run_mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> dict:
    """
    McNemar's test with continuity correction:
    Compares two classifiers on the exact same test set.
    b: Model A correct, Model B incorrect
    c: Model A incorrect, Model B correct
    chi2_stat = (|b - c| - 1)^2 / (b + c)
    """
    y_true = np.array(y_true)
    y_pred_a = np.array(y_pred_a)
    y_pred_b = np.array(y_pred_b)

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    n00 = np.sum(~correct_a & ~correct_b)
    n01 = np.sum(~correct_a & correct_b) # c: B correct, A incorrect
    n10 = np.sum(correct_a & ~correct_b) # b: A correct, B incorrect
    n11 = np.sum(correct_a & correct_b)

    b = int(n10)
    c = int(n01)
    
    if (b + c) == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = ((abs(b - c) - 1.0) ** 2) / (b + c)
        p_value = float(1.0 - chi2.cdf(statistic, df=1))

    is_significant = bool(p_value < 0.05)
    return {
        'b_modelA_correct_modelB_wrong': b,
        'c_modelA_wrong_modelB_correct': c,
        'both_correct': int(n11),
        'both_wrong': int(n00),
        'chi2_statistic': round(float(statistic), 4),
        'p_value': float(p_value),
        'p_value_formatted': f"{p_value:.6e}" if p_value < 1e-4 else f"{p_value:.4f}",
        'is_statistically_significant_at_05': is_significant
    }

def plot_confusion_matrix(y_true: list, y_pred: list, label_names: list[str], save_path: str, title: str = "Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)), normalize='true')
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=label_names, yticklabels=label_names, ax=ax, cbar=True)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()

def plot_training_curves(history: dict, save_path: str, model_name: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r--s', label='Val Loss', linewidth=2)
    ax1.set_title(f'{model_name} - Loss per Epoch', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.grid(True, linestyle=':')
    ax1.legend(fontsize=10)

    # Macro-F1
    ax2.plot(epochs, history['train_f1'], 'g-o', label='Train Macro-F1', linewidth=2)
    ax2.plot(epochs, history['val_f1'], 'm--s', label='Val Macro-F1', linewidth=2)
    ax2.set_title(f'{model_name} - Macro-F1 per Epoch', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Macro-F1 Score', fontsize=11)
    ax2.grid(True, linestyle=':')
    ax2.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()
