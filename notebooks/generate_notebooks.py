"""
Generates clean, fully-documented, standalone Jupyter Notebooks for all 8 phases.
Adheres strictly to the assignment rubric: reproducible, fixed random seeds,
clearly labeled with student metadata, explanations, and visualizations.
"""
import os
import nbformat as nbf

STUDENT_METADATA_MD = """# CS4045 - Deep Learning for Perception
## Assignment #3: Emotion Classification Using RNN, LSTM, GRU and Transformer
**Instructor:** Dr. Ahmad Raza Shahid  
**Student Name:** [Your Name]  
**Roll Number:** [i22-xxxx]  
**Reproducibility:** Seed = 42 | Device: Auto (CUDA if available, else CPU)  
---
"""

def create_phase1_nb():
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(STUDENT_METADATA_MD + """## Phase 1 – Preprocessing and Exploratory Analysis (15 points)
### Objectives:
1. Load GoEmotions (simplified configuration) dataset.
2. Prune multi-label and zero-label comments, retaining only single-label instances.
3. Clean and tokenize text, building vocabulary strictly on the training partition.
4. Analyze class frequencies and sequence length distributions; justify `max_len = 50`.
5. Identify hypothesized semantic confusions between overlapping emotional categories.
"""))
    
    nb.cells.append(nbf.v4.new_code_cell("""import os, sys
sys.path.insert(0, os.path.abspath('..'))
import json
import matplotlib.pyplot as plt
from src.data.eda_phase1 import run_phase1_eda

# Run exploratory data analysis and generate distribution plots
run_phase1_eda()
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""### Visualizations & Statistical Summaries
The generated figures display:
1. **Class Distribution**: Extreme 328:1 imbalance between majority class `neutral` (12,823 samples) and minority class `grief` (39 samples).
2. **Sequence Lengths**: Mean = 15.37, 95th percentile = 29, 99th percentile = 32 tokens. A cutoff of `max_len = 50` covers >99.5% of comments.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""from IPython.display import Image, display

display(Image('../experiments/figures/phase1_class_distribution.png'))
display(Image('../experiments/figures/phase1_sequence_lengths.png'))

with open('../experiments/metrics/phase1_eda_summary.json') as f:
    summary = json.load(f)
print("Filtering Stats:", json.dumps(summary['filtering_stats'], indent=2))
print("OOV Rates: Val =", summary['val_oov_pct'], "%, Test =", summary['test_oov_pct'], "%")
"""))
    return nb

def create_phase2_nb():
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(STUDENT_METADATA_MD + """## Phase 2 – Recurrent Baselines: RNN, LSTM, GRU (20 points)
### Objectives:
1. Standardized embedding setup ($d_e = 100, d_h = 128$).
2. Fair tuning under identical learning rates and regularization.
3. Track training/validation loss and macro-F1 trajectories with early stopping on validation split.
4. Report test accuracy, macro-F1, and per-class F1 scores.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""import os, sys
sys.path.insert(0, os.path.abspath('..'))
import json
from src.run_phase2_baselines import run_phase2

# Execute training and evaluation for Simple RNN, LSTM, and GRU
run_phase2()
"""))

    nb.cells.append(nbf.v4.new_code_cell("""from IPython.display import Image, display
display(Image('../experiments/figures/phase2_training_curves.png'))

with open('../experiments/metrics/phase2_recurrent_baselines.json') as f:
    res = json.load(f)
for model, data in res.items():
    print(f"=== {model} ===")
    print(f"Parameters: {data['parameters']['total_parameters']:,}")
    print(f"Test Accuracy: {data['test_metrics']['accuracy']:.4f}")
    print(f"Test Macro-F1: {data['test_metrics']['macro_f1']:.4f}\\n")
"""))
    return nb

def create_phase3_nb():
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(STUDENT_METADATA_MD + """## Phase 3 – Transformer Encoder from Scratch (25 points)
### Objectives:
1. Pure scratch implementation of Multi-Head Self-Attention ($h=4, d_k=25, d_{model}=100$).
2. Sinusoidal positional encodings, Pre-LN residual blocks, position-wise feed-forward ($d_{ff}=256$), and classification head.
3. Strictly NO `nn.MultiheadAttention`, `nn.TransformerEncoder`, or pretrained models.
4. Tune and report test accuracy and macro-F1.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""import os, sys
sys.path.insert(0, os.path.abspath('..'))
from src.run_phase3_transformer import run_phase3

run_phase3()
"""))

    nb.cells.append(nbf.v4.new_code_cell("""from IPython.display import Image, display
import json

display(Image('../experiments/figures/phase3_transformer_curves.png'))
with open('../experiments/metrics/phase3_transformer.json') as f:
    data = json.load(f)
print(f"Scratch Transformer Total Parameters: {data['parameters']['total_parameters']:,}")
print(f"Test Accuracy: {data['test_metrics']['accuracy']:.4f}")
print(f"Test Macro-F1: {data['test_metrics']['macro_f1']:.4f}")
"""))
    return nb

def create_phase4_nb():
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(STUDENT_METADATA_MD + """## Phase 4 – Class Imbalance and Evaluation Metrics (15 points)
### Research Question:
Does accuracy hide poor performance on small classes, and which method works best?
### Objectives:
1. Compare standard unweighted Cross-Entropy vs Class-Weighted Cross-Entropy vs Focal Loss ($\gamma=2.0$).
2. Evaluate Macro-F1, minority class recall (`grief`, `pride`, `nervousness`, `relief`, `embarrassment`), and confusion matrices.
3. Select best strategy and retrain all 4 canonical models (RNN, LSTM, GRU, Transformer) for use in subsequent phases.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""import os, sys
sys.path.insert(0, os.path.abspath('..'))
from src.run_phase4_imbalance import run_phase4

run_phase4()
"""))

    nb.cells.append(nbf.v4.new_code_cell("""from IPython.display import Image, display
import json

display(Image('../experiments/figures/phase4_minority_recall_comparison.png'))
display(Image('../experiments/figures/phase4_cm_class-weighted_ce.png'))

with open('../experiments/metrics/phase4_imbalance_comparison.json') as f:
    data = json.load(f)
print("Best Selected Strategy:", data['selected_best_strategy'])
print("\\nStrategies Comparison:")
print(json.dumps(data['loss_strategies_compared'], indent=2))
"""))
    return nb

def create_phase5_nb():
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(STUDENT_METADATA_MD + """## Phase 5 – Sequence Readout and Direction (15 points)
### Research Question:
How should a sequence be turned into one vector, and does reading in both directions help? Use the LSTM.
### Objectives:
1. Compare 3 readout mechanisms: Last Hidden State, Average Pooling, Max Pooling.
2. Compare 2 reading directions: Forward-only vs Bidirectional.
3. Evaluate full 6-configuration matrix on Accuracy, Macro-F1, and Inference Latency.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""import os, sys
sys.path.insert(0, os.path.abspath('..'))
from src.run_phase5_readout import run_phase5

run_phase5()
"""))

    nb.cells.append(nbf.v4.new_code_cell("""from IPython.display import Image, display
import json

display(Image('../experiments/figures/phase5_readout_direction.png'))
with open('../experiments/metrics/phase5_readout_direction.json') as f:
    data = json.load(f)
for cfg, val in data.items():
    print(f"{cfg:<20}: Acc = {val['test_accuracy']:.4f} | Macro-F1 = {val['test_macro_f1']:.4f} | Latency = {val['latency']['batch_latency_mean_ms']:.2f} ms")
"""))
    return nb

def create_phase6_nb():
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(STUDENT_METADATA_MD + """## Phase 6 – Word Representation: Random vs Pretrained Embeddings (15 points)
### Research Question:
How much does pretrained knowledge help? Use the four models from Phase 4.
### Objectives:
1. Match vocabulary against GloVe 100d embeddings; report coverage percentage.
2. Compare Random Embeddings vs Pretrained GloVe (Frozen) vs Pretrained GloVe (Fine-tuned).
3. Identify which architectures and emotion classes gain most from pretrained representations.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""import os, sys
sys.path.insert(0, os.path.abspath('..'))
from src.run_phase6_embeddings import run_phase6

run_phase6()
"""))

    nb.cells.append(nbf.v4.new_code_cell("""from IPython.display import Image, display
import json

display(Image('../experiments/figures/phase6_embedding_comparison.png'))
with open('../experiments/metrics/phase6_embeddings.json') as f:
    data = json.load(f)
print("GloVe Coverage:", data['coverage_info']['coverage_pct'], "%")
print("Sample missing slang/abbreviations:", data['coverage_info']['sample_missing'])
"""))
    return nb

def create_phase7_nb():
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(STUDENT_METADATA_MD + """## Phase 7 – Attention Analysis (15 points)
### Research Question:
Do the words a model attends to actually decide its prediction?
### Objectives:
1. Extract and visualize multi-head attention maps for correctly and wrongly classified comments.
2. Conduct token importance ablation experiment: compare accuracy drop when masking top-attended tokens vs random tokens vs bottom-attended tokens.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""import os, sys
sys.path.insert(0, os.path.abspath('..'))
from src.run_phase7_attention import run_phase7

run_phase7()
"""))

    nb.cells.append(nbf.v4.new_code_cell("""from IPython.display import Image, display
import json

display(Image('../experiments/figures/phase7_attention_heatmaps.png'))
display(Image('../experiments/figures/phase7_ablation_fidelity.png'))

with open('../experiments/metrics/phase7_attention_analysis.json') as f:
    data = json.load(f)
print("Ablation Results Summary:")
print(json.dumps(data['ablation_experiment'], indent=2))
"""))
    return nb

def create_phase8_nb():
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(STUDENT_METADATA_MD + """## Phase 8 – Comparison and Robustness (15 points)
### Research Questions:
Which model is best overall, and which is the most robust to noisy text?
### Objectives:
1. Benchmark 4 canonical models on Accuracy, Macro-F1, Parameter Count, Size, and Latency.
2. Run McNemar's statistical significance test between Best Recurrent Model and Scratch Transformer.
3. Design zero-shot test-time noise experiment (0% to 30% typo/swap/deletion corruption) and compare degradation curves.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""import os, sys
sys.path.insert(0, os.path.abspath('..'))
from src.run_phase8_robustness import run_phase8

run_phase8()
"""))

    nb.cells.append(nbf.v4.new_code_cell("""from IPython.display import Image, display
import json

display(Image('../experiments/figures/phase8_noise_robustness.png'))
with open('../experiments/metrics/phase8_benchmark_robustness.json') as f:
    data = json.load(f)

print("McNemar's Statistical Test:")
print(json.dumps(data['mcnemar_statistical_test'], indent=2))

print("\\nBenchmark Table:")
print(json.dumps(data['benchmark_comparison_table'], indent=2))
"""))
    return nb

def create_master_nb():
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(STUDENT_METADATA_MD + """# Master Notebook: All Phases (1 to 8)
This notebook sequentially coordinates all eight phases of Assignment #3.
"""))
    nb.cells.append(nbf.v4.new_code_cell("""import os, sys
sys.path.insert(0, os.path.abspath('..'))
from src.data.eda_phase1 import run_phase1_eda
from src.run_phase2_baselines import run_phase2
from src.run_phase3_transformer import run_phase3
from src.run_phase4_imbalance import run_phase4
from src.run_phase5_readout import run_phase5
from src.run_phase6_embeddings import run_phase6
from src.run_phase7_attention import run_phase7
from src.run_phase8_robustness import run_phase8

print("=== Running Phase 1 ===")
run_phase1_eda()
print("=== Running Phase 2 ===")
run_phase2()
print("=== Running Phase 3 ===")
run_phase3()
print("=== Running Phase 4 ===")
run_phase4()
print("=== Running Phase 5 ===")
run_phase5()
print("=== Running Phase 6 ===")
run_phase6()
print("=== Running Phase 7 ===")
run_phase7()
print("=== Running Phase 8 ===")
run_phase8()
print("=== ALL PHASES COMPLETED SUCCESSFULLY! ===")
"""))
    return nb

def build_all_notebooks():
    os.makedirs('notebooks', exist_ok=True)
    nb_map = {
        'notebooks/01_Phase1_Exploratory_Data_Analysis.ipynb': create_phase1_nb(),
        'notebooks/02_Phase2_Recurrent_Baselines.ipynb': create_phase2_nb(),
        'notebooks/03_Phase3_Transformer_From_Scratch.ipynb': create_phase3_nb(),
        'notebooks/04_Phase4_Class_Imbalance_Solutions.ipynb': create_phase4_nb(),
        'notebooks/05_Phase5_Sequence_Readout_and_Direction.ipynb': create_phase5_nb(),
        'notebooks/06_Phase6_Pretrained_GloVe_Embeddings.ipynb': create_phase6_nb(),
        'notebooks/07_Phase7_Attention_Analysis.ipynb': create_phase7_nb(),
        'notebooks/08_Phase8_Benchmarking_and_Robustness.ipynb': create_phase8_nb(),
        'notebooks/CS4045_Assignment3_Master.ipynb': create_master_nb()
    }
    for path, nb in nb_map.items():
        with open(path, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        print(f"Generated notebook: {path}")

if __name__ == '__main__':
    build_all_notebooks()
