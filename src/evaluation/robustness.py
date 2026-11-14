"""
Phase 8: Test-Time Noise Robustness Experiment.
Evaluates model degradation under noisy inputs (typos, deletions, insertions) WITHOUT retraining.
"""
import random
import string
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from src.data.dataset import tokenize_text, GoEmotionsDataset

def inject_noise_into_text(text: str, noise_rate: float = 0.1) -> str:
    """
    Inject realistic Reddit-style textual noise:
    - Character swaps (adjacent transposed letters)
    - Character drops (omitted keys)
    - Character substitutions (nearby keyboard typos)
    """
    if noise_rate <= 0:
        return text

    words = text.split()
    noisy_words = []

    for word in words:
        if random.random() < noise_rate and len(word) > 2:
            op = random.choice(['swap', 'drop', 'sub'])
            chars = list(word)
            idx = random.randint(0, len(chars) - 2)
            if op == 'swap':
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            elif op == 'drop':
                del chars[idx]
            elif op == 'sub':
                chars[idx] = random.choice(string.ascii_lowercase)
            noisy_words.append("".join(chars))
        else:
            noisy_words.append(word)

    return " ".join(noisy_words)

def evaluate_on_noisy_data(model, test_ds, vocab, label_names: list[str], device: torch.device, noise_rates: list[float] = [0.0, 0.05, 0.1, 0.2, 0.3], batch_size: int = 64) -> dict:
    model.eval()
    results = {'noise_rates': noise_rates, 'accuracies': [], 'macro_f1s': []}

    for rate in noise_rates:
        # Create noisy version of the test items
        noisy_items = []
        for orig in test_ds.items:
            noisy_text = inject_noise_into_text(orig['text'], noise_rate=rate)
            noisy_items.append({'text': noisy_text, 'label': orig['label'], 'id': orig.get('id', '')})

        noisy_dataset = GoEmotionsDataset(noisy_items, vocab, max_len=test_ds.max_len)
        loader = DataLoader(noisy_dataset, batch_size=batch_size, shuffle=False)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                lengths = batch['length'].to(device)
                targets = batch['label'].to(device)

                logits = model(input_ids, lengths)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy())

        acc = float(accuracy_score(all_targets, all_preds))
        f1 = float(f1_score(all_targets, all_preds, average='macro', zero_division=0))

        results['accuracies'].append(round(acc, 4))
        results['macro_f1s'].append(round(f1, 4))
        print(f"Noise Rate: {rate*100:.0f}% -> Accuracy: {acc:.4f}, Macro-F1: {f1:.4f}")

    return results

def plot_noise_robustness(comparison_results: dict, save_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    rates = [r * 100 for r in comparison_results['noise_rates']]

    styles = {
        'Simple RNN': {'color': 'orange', 'marker': 'o', 'linestyle': '--'},
        'LSTM': {'color': 'blue', 'marker': 's', 'linestyle': '-'},
        'GRU': {'color': 'green', 'marker': '^', 'linestyle': '-.'},
        'Transformer': {'color': 'crimson', 'marker': 'D', 'linestyle': '-'}
    }

    for name, data in comparison_results['models'].items():
        st = styles.get(name, {'color': 'black', 'marker': 'o', 'linestyle': '-'})
        ax1.plot(rates, data['accuracies'], label=name, color=st['color'], marker=st['marker'], linestyle=st['linestyle'], linewidth=2)
        ax2.plot(rates, data['macro_f1s'], label=name, color=st['color'], marker=st['marker'], linestyle=st['linestyle'], linewidth=2)

    ax1.set_title('Accuracy vs Input Noise Level', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Noise Rate (%)', fontsize=11)
    ax1.set_ylabel('Test Accuracy', fontsize=11)
    ax1.grid(True, linestyle=':')
    ax1.legend(fontsize=10)

    ax2.set_title('Macro-F1 vs Input Noise Level', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Noise Rate (%)', fontsize=11)
    ax2.set_ylabel('Test Macro-F1', fontsize=11)
    ax2.grid(True, linestyle=':')
    ax2.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()
