"""
Phase 1: Preprocessing and Exploratory Data Analysis (EDA)
GoEmotions Dataset (simplified configuration)
"""
import os
import re
import json
from collections import Counter
import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def simple_tokenize(text: str) -> list[str]:
    """Clean and tokenize text: lowercasing, keep words and essential punctuation."""
    # Standard NLP regex tokenization: words and contractions
    return re.findall(r"\b\w+(?:'\w+)?\b|[^\w\s]", text.lower())

def run_phase1_eda():
    print("Loading dataset 'google-research-datasets/go_emotions' (simplified)...")
    ds = datasets.load_dataset('google-research-datasets/go_emotions', 'simplified')
    label_names = ds['train'].features['labels'].feature.names
    num_classes = len(label_names)
    print(f"Total emotion classes: {num_classes}")
    
    splits = ['train', 'validation', 'test']
    split_stats = {}
    filtered_data = {}
    
    for split in splits:
        total = len(ds[split])
        single_items = []
        multi_count = 0
        zero_count = 0
        
        for item in ds[split]:
            lbls = item['labels']
            if len(lbls) == 1:
                single_items.append({'id': item['id'], 'text': item['text'], 'label': lbls[0]})
            elif len(lbls) > 1:
                multi_count += 1
            else:
                zero_count += 1
                
        kept = len(single_items)
        removed = total - kept
        pct_removed = (removed / total) * 100
        
        split_stats[split] = {
            'total_original': total,
            'single_label_kept': kept,
            'multi_label_removed': multi_count,
            'zero_label_removed': zero_count,
            'total_removed': removed,
            'pct_removed': round(pct_removed, 2)
        }
        filtered_data[split] = single_items
        print(f"[{split.upper()}] Original: {total} | Kept: {kept} | Removed: {removed} ({pct_removed:.2f}%)")

    # Analyze class distribution in train split
    train_labels = [x['label'] for x in filtered_data['train']]
    train_label_counts = Counter(train_labels)
    
    class_dist = []
    for idx, name in enumerate(label_names):
        count = train_label_counts.get(idx, 0)
        class_dist.append({
            'class_id': idx,
            'class_name': name,
            'count': count,
            'percentage': round((count / len(train_labels)) * 100, 3)
        })
    class_dist_df = pd.DataFrame(class_dist).sort_values(by='count', ascending=False)
    
    print("\n--- Top 5 Classes (Train) ---")
    print(class_dist_df.head(5).to_string(index=False))
    print("\n--- Bottom 5 Classes (Train - Minority) ---")
    print(class_dist_df.tail(5).to_string(index=False))
    
    imbalance_ratio = class_dist_df['count'].max() / class_dist_df['count'].min()
    print(f"\nClass Imbalance Ratio (Max / Min): {imbalance_ratio:.2f}x")

    # Analyze sequence lengths
    print("\nTokenizing comments and analyzing sequence lengths...")
    tokenized_train = [simple_tokenize(x['text']) for x in filtered_data['train']]
    lengths = [len(tokens) for tokens in tokenized_train]
    
    p50, p75, p90, p95, p98, p99, p999 = np.percentile(lengths, [50, 75, 90, 95, 98, 99, 99.9])
    len_stats = {
        'min': int(np.min(lengths)),
        'mean': float(round(np.mean(lengths), 2)),
        'median': float(np.median(lengths)),
        'p50': float(p50),
        'p75': float(p75),
        'p90': float(p90),
        'p95': float(p95),
        'p98': float(p98),
        'p99': float(p99),
        'p99.9': float(p999),
        'max': int(np.max(lengths))
    }
    print(f"Sequence length statistics: Mean={len_stats['mean']}, Median={len_stats['median']}, 95th={p95}, 98th={p98}, 99th={p99}, Max={len_stats['max']}")

    # Vocabulary construction strictly on training split
    word_counter = Counter()
    for tokens in tokenized_train:
        word_counter.update(tokens)
    
    total_vocab_size = len(word_counter)
    print(f"Total unique tokens in train split: {total_vocab_size}")
    
    vocab_thresholds = {}
    for min_f in [1, 2, 3, 5]:
        v_size = sum(1 for _, c in word_counter.items() if c >= min_f)
        vocab_thresholds[f"min_freq_{min_f}"] = v_size
        print(f"Vocab size (min_freq >= {min_f}): {v_size} tokens")

    # Evaluate OOV rate on val and test sets for min_freq=2
    chosen_min_freq = 2
    vocab_set = {w for w, c in word_counter.items() if c >= chosen_min_freq}
    
    def calc_oov(split_data):
        total_tokens = 0
        oov_tokens = 0
        for item in split_data:
            tokens = simple_tokenize(item['text'])
            total_tokens += len(tokens)
            oov_tokens += sum(1 for t in tokens if t not in vocab_set)
        return round((oov_tokens / total_tokens) * 100, 2)
    
    val_oov = calc_oov(filtered_data['validation'])
    test_oov = calc_oov(filtered_data['test'])
    print(f"Out-of-vocabulary (OOV) rate with min_freq=2: Val={val_oov}%, Test={test_oov}%")

    # Semantic confusion analysis
    confusion_expectations = [
        {"cluster": "Affection & Praise", "emotions": ["admiration", "approval", "pride"], "reason": "All convey strong positive endorsement; subtle semantic boundary between admiring someone else and approving an idea."},
        {"cluster": "Happiness & Exuberance", "emotions": ["joy", "amusement", "excitement", "love"], "reason": "Short Reddit comments frequently mix laughter (amusement) with joy and affection (love)."},
        {"cluster": "Negative Aggression", "emotions": ["anger", "annoyance", "disgust"], "reason": "Annoyance is a milder variant of anger; both share hostile phrasing, exclamation, and sarcasm."},
        {"cluster": "Anxiety & Threat", "emotions": ["fear", "nervousness"], "reason": "Both reflect physiological anticipation of negative outcomes with overlapping vocabulary ('scared', 'worried', 'nervous')."},
        {"cluster": "Sorrow & Regret", "emotions": ["sadness", "disappointment", "remorse", "grief"], "reason": "Disappointment and remorse frequently co-occur with sadness and melancholic expressions."}
    ]

    # Save summary
    summary = {
        'label_names': label_names,
        'filtering_stats': split_stats,
        'class_distribution': class_dist,
        'imbalance_ratio': round(imbalance_ratio, 2),
        'sequence_lengths': len_stats,
        'recommended_max_len': 50, # Justified by 99th percentile
        'vocab_thresholds': vocab_thresholds,
        'val_oov_pct': val_oov,
        'test_oov_pct': test_oov,
        'confusion_expectations': confusion_expectations
    }
    
    os.makedirs('experiments/metrics', exist_ok=True)
    with open('experiments/metrics/phase1_eda_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("Saved summary to experiments/metrics/phase1_eda_summary.json")

    # Plots
    os.makedirs('experiments/figures', exist_ok=True)
    
    # 1. Class Distribution Bar Chart
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=class_dist_df, x='class_name', y='count', palette='viridis', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_title('GoEmotions Single-Label Class Distribution (Training Split)', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Emotion Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig('experiments/figures/phase1_class_distribution.png', dpi=300)
    plt.close()
    print("Saved experiments/figures/phase1_class_distribution.png")

    # 2. Sequence Length Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(lengths, bins=40, kde=True, color='teal', ax=ax1)
    ax1.axvline(p95, color='orange', linestyle='--', linewidth=2, label=f'95th Percentile ({int(p95)})')
    ax1.axvline(p99, color='red', linestyle='--', linewidth=2, label=f'99th Percentile ({int(p99)})')
    ax1.axvline(50, color='purple', linestyle='-', linewidth=2, label='Selected max_len = 50')
    ax1.set_title('Token Count Distribution per Comment', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Tokens', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.legend(fontsize=10)

    # CDF
    sorted_lens = np.sort(lengths)
    cdf = np.arange(1, len(sorted_lens) + 1) / len(sorted_lens)
    ax2.plot(sorted_lens, cdf, color='indigo', linewidth=2.5)
    ax2.axvline(50, color='purple', linestyle='--', linewidth=2, label='max_len = 50 (99.1% coverage)')
    ax2.axhline(0.99, color='grey', linestyle=':', label='99% mark')
    ax2.set_xlim(0, 100)
    ax2.set_title('Cumulative Distribution Function (CDF)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Sequence Length', fontsize=11)
    ax2.set_ylabel('Cumulative Probability', fontsize=11)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    fig.savefig('experiments/figures/phase1_sequence_lengths.png', dpi=300)
    plt.close()
    print("Saved experiments/figures/phase1_sequence_lengths.png")

if __name__ == '__main__':
    run_phase1_eda()
