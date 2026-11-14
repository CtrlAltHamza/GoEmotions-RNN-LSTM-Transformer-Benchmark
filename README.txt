================================================================================
Emotion Classification Benchmark on GoEmotions
Comparative Experimental Study: RNN, LSTM, GRU, and Scratch Transformer
================================================================================

1. PROJECT OVERVIEW
--------------------------------------------------------------------------------
This project implements a reproducible empirical benchmark on fine-grained emotion 
classification using the GoEmotions dataset (simplified split, 28 classes).
- Strict zero-leakage preprocessing; vocabulary constructed purely from train split.
- Recurrent baselines (Simple RNN, LSTM, GRU) under standardized parameter budgets.
- Pure Scratch Transformer Encoder (Multi-Head Self-Attention, Sinusoidal Positional 
  Encoding, Pre-LN residuals, Feed-Forward block, Classification head) without banned 
  high-level abstractions.
- Mitigation of 328:1 class skew via Class-Weighted Cross-Entropy and Focal Loss.
- Sequence readout mechanisms (Last, Mean, Max pooling) and direction study on LSTM.
- Pretrained GloVe 100d transfer learning analysis (Frozen vs Fine-tuned).
- Attention interpretability heatmaps and token-importance ablation fidelity tests.
- Model profiling, McNemar statistical significance testing (p < 0.001), and 
  zero-shot typographical noise degradation (0% to 30%).

2. DIRECTORY STRUCTURE
--------------------------------------------------------------------------------
├── notebooks/                              # Jupyter Notebooks for all phases (01 to 08)
├── src/                                    # Modular Python pipeline
│   ├── data/                               # Dataset loader, vocab & EDA
│   ├── models/                             # RNN, LSTM, GRU, Transformer, GloVe
│   ├── training/                           # Losses & trainer loop
│   ├── evaluation/                         # Metrics, attention, robustness
│   └── utils/                              # Reproducibility, profiler, tracking
├── experiments/
│   ├── figures/                            # 10 High-resolution 300 DPI plots
│   ├── metrics/                            # Metric summaries (JSON)
│   └── checkpoints/                        # Model weights (.pt)
├── report/
│   ├── report.tex                          # Formal LaTeX paper
│   └── build_report_pdf.py                 # Automated PDF builder
├── report.pdf                              # Formal Research Report (PDF)
├── README.txt                              # Standard text reference
└── README.md                               # Markdown documentation for GitHub

3. HOW TO RUN
--------------------------------------------------------------------------------
Install dependencies:
    pip install torch datasets scikit-learn pandas numpy scipy matplotlib seaborn mlflow fpdf2

Execute any phase directly from the root directory:
    python src/data/eda_phase1.py
    python src/run_phase2_baselines.py
    python src/run_phase3_transformer.py
    python src/run_phase4_imbalance.py
    python src/run_phase5_readout.py
    python src/run_phase6_embeddings.py
    python src/run_phase7_attention.py
    python src/run_phase8_robustness.py

Interactive Notebooks:
    Launch Jupyter Lab or Notebook and navigate to the notebooks/ folder.
================================================================================
