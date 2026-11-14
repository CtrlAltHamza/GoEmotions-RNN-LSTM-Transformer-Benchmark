"""
Phase 7: Attention Interpretability and Ablation Experiment.
Visualizes multi-head attention on correct and incorrect predictions.
Executes token importance ablation testing top-attended vs random vs bottom-attended words.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import torch
from src.data.dataset import get_go_emotions_data, tokenize_text
from src.models.transformer import TransformerClassifierFromScratch
from src.evaluation.attention_analysis import extract_token_attentions, plot_attention_heatmaps, run_attention_ablation_experiment, plot_ablation_results
from src.utils.reproducibility import set_seed, get_device
from src.utils.mlops_tracker import MLOpsTracker

def run_phase7():
    set_seed(42)
    torch.set_num_threads(4)
    device = get_device()
    tracker = MLOpsTracker()
    tracker.start_run(run_name="Phase7_Attention_Analysis", phase="Phase7")

    print("\nLoading dataset for Phase 7...")
    data = get_go_emotions_data(batch_size=128)
    vocab = data['vocab']
    label_names = data['label_names']
    num_classes = len(label_names)
    vocab_size = len(vocab)
    test_ds = data['test_ds']

    # Initialize scratch Transformer and load canonical checkpoint
    model = TransformerClassifierFromScratch(
        vocab_size=vocab_size,
        embedding_dim=100,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        num_classes=num_classes
    ).to(device)

    ckpt_path = "experiments/checkpoints/canonical_transformer.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "experiments/checkpoints/phase3_transformer_scratch.pt"
    
    print(f"Loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Find 2 correct and 2 misclassified examples for visualization
    correct_samples = []
    incorrect_samples = []

    with torch.no_grad():
        for i in range(len(test_ds)):
            item = test_ds[i]
            inp = item['input_ids'].unsqueeze(0).to(device)
            target = item['label'].item()
            length = item['length'].item()
            text = test_ds.items[i]['text']
            tokens = tokenize_text(text)[:length]

            # We want interesting comments of reasonable length (5 to 15 tokens)
            if length < 5 or length > 15:
                continue

            logits, attn_weights, token_scores = extract_token_attentions(model, inp, device)
            pred = torch.argmax(logits, dim=-1).item()
            prob = torch.softmax(logits, dim=-1)[0, pred].item()

            rec = {
                'text': text,
                'tokens': tokens,
                'true_label': label_names[target],
                'pred_label': label_names[pred],
                'prob': round(float(prob), 3),
                'correct': bool(pred == target),
                'attention_matrix': attn_weights[0].mean(dim=0).cpu().numpy() # [seq_len, seq_len]
            }

            if pred == target and prob > 0.6 and len(correct_samples) < 2:
                correct_samples.append(rec)
            elif pred != target and len(incorrect_samples) < 2:
                incorrect_samples.append(rec)

            if len(correct_samples) == 2 and len(incorrect_samples) == 2:
                break

    selected_records = correct_samples + incorrect_samples
    print("\nSelected Samples for Attention Heatmap Visualization:")
    for r in selected_records:
        tag = "CORRECT" if r['correct'] else "WRONG"
        print(f"[{tag}] '{r['text']}' | True: {r['true_label']} -> Pred: {r['pred_label']} ({r['prob']})")

    # Plot Heatmaps
    plot_attention_heatmaps(selected_records, 'experiments/figures/phase7_attention_heatmaps.png')
    print("Saved experiments/figures/phase7_attention_heatmaps.png")

    # Run Token Importance Ablation Experiment
    print("\nRunning Token Importance Ablation Experiment...")
    ablation_results = run_attention_ablation_experiment(
        model=model,
        test_ds=test_ds,
        vocab=vocab,
        label_names=label_names,
        device=device,
        num_samples=500,
        k_list=[1, 2, 3]
    )

    plot_ablation_results(ablation_results, 'experiments/figures/phase7_ablation_fidelity.png')
    print("Saved experiments/figures/phase7_ablation_fidelity.png")

    # Save summary
    summary = {
        'visualized_samples': [
            {
                'text': r['text'],
                'tokens': r['tokens'],
                'true_label': r['true_label'],
                'pred_label': r['pred_label'],
                'confidence': r['prob'],
                'is_correct': r['correct']
            }
            for r in selected_records
        ],
        'ablation_experiment': ablation_results
    }

    with open('experiments/metrics/phase7_attention_analysis.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("Saved experiments/metrics/phase7_attention_analysis.json")
    tracker.end_run()

if __name__ == '__main__':
    run_phase7()
