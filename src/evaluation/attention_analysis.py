"""
Phase 7: Attention Visualization and Token Ablation/Importance Experiment.
Tests whether the most attended tokens in the Transformer Encoder decide predictions.
"""
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def extract_token_attentions(model, input_ids: torch.Tensor, device: torch.device):
    model.eval()
    with torch.no_grad():
        logits, attn_weights = model(input_ids.to(device), return_attention=True)
    # attn_weights shape: [batch_size, num_heads, seq_len, seq_len]
    # Average across all heads
    mean_head_attn = attn_weights.mean(dim=1) # [batch_size, seq_len, seq_len]
    # Token importance: average attention received across sequence positions
    token_importance = mean_head_attn.mean(dim=1) # [batch_size, seq_len]
    return logits, attn_weights, token_importance

def plot_attention_heatmaps(sample_records: list[dict], save_path: str):
    """
    Plots attention heatmaps for correctly and wrongly classified examples.
    """
    num_samples = len(sample_records)
    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 3.2 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for idx, rec in enumerate(sample_records):
        ax = axes[idx]
        tokens = rec['tokens']
        # attention matrix: [seq_len, seq_len]
        attn_mat = rec['attention_matrix'][:len(tokens), :len(tokens)]

        sns.heatmap(
            attn_mat,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='YlOrRd',
            ax=ax,
            cbar=True
        )
        status_tag = "CORRECT" if rec['correct'] else "WRONGLY CLASSIFIED"
        title_str = (
            f"[{status_tag}] \"{rec['text']}\"\n"
            f"True: '{rec['true_label']}' | Pred: '{rec['pred_label']}' (Confidence: {rec['prob']:.2f})"
        )
        ax.set_title(title_str, fontsize=11, fontweight='bold', pad=8)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()

def run_attention_ablation_experiment(
    model,
    test_ds,
    vocab,
    label_names: list[str],
    device: torch.device,
    num_samples: int = 500,
    k_list: list[int] = [1, 2, 3]
) -> dict:
    """
    Controlled ablation experiment:
    Compares the accuracy degradation when masking top-k attended words vs random k words vs bottom-k words.
    """
    model.eval()
    pad_idx = vocab.pad_idx
    unk_idx = vocab.unk_idx

    # Collect correctly classified samples
    candidates = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            item = test_ds[i]
            inp = item['input_ids'].unsqueeze(0).to(device)
            target = item['label'].item()
            length = item['length']
            if length < 4: # ensure enough tokens for k=3 ablation
                continue

            logits, attn_weights, token_scores = extract_token_attentions(model, inp, device)
            pred = torch.argmax(logits, dim=-1).item()
            prob = torch.softmax(logits, dim=-1)[0, pred].item()

            if pred == target:
                # Valid token scores (excluding padding)
                scores = token_scores[0, :length].cpu().numpy()
                candidates.append({
                    'input_ids': item['input_ids'].clone(),
                    'length': length,
                    'target': target,
                    'scores': scores,
                    'orig_prob': prob
                })
            if len(candidates) >= num_samples:
                break

    print(f"Running attention ablation on {len(candidates)} verified correct test samples...")

    results = {'k': k_list, 'top_attended': [], 'random_words': [], 'bottom_attended': []}

    for k in k_list:
        top_correct = 0
        rand_correct = 0
        bot_correct = 0

        for item in candidates:
            length = item['length']
            target = item['target']
            scores = item['scores']
            valid_indices = list(range(length))

            # 1. Top-k attended indices
            top_k_indices = np.argsort(scores)[-k:]

            # 2. Bottom-k attended indices
            bot_k_indices = np.argsort(scores)[:k]

            # 3. Random k indices (averaged across 3 random seeds for stability)
            rand_trials = []
            for _ in range(3):
                rand_indices = random.sample(valid_indices, k)
                inp_rand = item['input_ids'].clone()
                for idx in rand_indices:
                    inp_rand[idx] = unk_idx
                with torch.no_grad():
                    out_rand = model(inp_rand.unsqueeze(0).to(device))
                    rand_trials.append(1 if torch.argmax(out_rand, dim=-1).item() == target else 0)

            # Test top-k ablation
            inp_top = item['input_ids'].clone()
            for idx in top_k_indices:
                inp_top[idx] = unk_idx
            with torch.no_grad():
                out_top = model(inp_top.unsqueeze(0).to(device))
                if torch.argmax(out_top, dim=-1).item() == target:
                    top_correct += 1

            # Test bottom-k ablation
            inp_bot = item['input_ids'].clone()
            for idx in bot_k_indices:
                inp_bot[idx] = unk_idx
            with torch.no_grad():
                out_bot = model(inp_bot.unsqueeze(0).to(device))
                if torch.argmax(out_bot, dim=-1).item() == target:
                    bot_correct += 1

            rand_correct += np.mean(rand_trials)

        n = len(candidates)
        top_acc = (top_correct / n) * 100.0
        rand_acc = (rand_correct / n) * 100.0
        bot_acc = (bot_correct / n) * 100.0

        results['top_attended'].append(round(top_acc, 2))
        results['random_words'].append(round(rand_acc, 2))
        results['bottom_attended'].append(round(bot_acc, 2))

        print(f"k={k} Masked | Top-Attended Acc: {top_acc:.2f}% | Random Acc: {rand_acc:.2f}% | Bottom-Attended Acc: {bot_acc:.2f}%")

    return results

def plot_ablation_results(results: dict, save_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    k = results['k']

    ax.plot(k, results['top_attended'], 'r-o', linewidth=2.5, markersize=8, label='Top-Attended Tokens Masked')
    ax.plot(k, results['random_words'], 'b--s', linewidth=2, markersize=7, label='Random Tokens Masked')
    ax.plot(k, results['bottom_attended'], 'g-.^', linewidth=2, markersize=7, label='Bottom-Attended Tokens Masked')

    ax.set_title('Attention Fidelity Test: Impact of Token Ablation', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Number of Tokens Masked (k)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Model Accuracy Remaining (%)', fontsize=11, fontweight='bold')
    ax.set_xticks(k)
    ax.grid(True, linestyle=':')
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()
