================================================================================
CS4045 - Deep Learning for Perception (Fall 2026)
Assignment #3: Emotion Classification Using RNN, LSTM, GRU and Transformer
Instructor: Dr. Ahmad Raza Shahid
--------------------------------------------------------------------------------
Student Name: Hamza Khurram
Roll Number:  i210735
Submission:   i210735_Hamza_Khurram_A3.zip
================================================================================

1. PROJECT OVERVIEW
--------------------------------------------------------------------------------
This project implements an end-to-end, research-grade experimental study on fine-
grained emotion classification using the GoEmotions dataset (simplified configuration,
28 classes). It strictly adheres to all assignment guidelines and MLOps best practices:
- Zero data leakage: Tokenizer vocabulary and thresholds are built exclusively on 
  the training partition.
- Validation split is used for all tuning decisions, early stopping, and model selection.
- Test split is used exclusively for final reporting.
- A pure scratch implementation of the Transformer Encoder (Multi-Head Self-Attention,
  Positional Encodings, Pre-LN Residuals, Feed-Forward Block, Classification Head) 
  is developed without nn.MultiheadAttention or any pretrained transformers.
- Comprehensive mitigation of extreme 328:1 class imbalance via Class-Weighted 
  Cross-Entropy and Multi-Class Focal Loss.
- Systematic sequence readout (Last vs Mean vs Max pooling) and direction study on LSTM.
- Pretrained GloVe 100d word representation transfer (Frozen vs Fine-tuned).
- Transformer attention interpretability and token ablation fidelity study.
- Model benchmarking, McNemar statistical significance testing (p < 0.001), and 
  test-time noise robustness (0% to 30% typo/swap/deletion degradation).

2. DIRECTORY STRUCTURE
--------------------------------------------------------------------------------
.
├── notebooks/                              # Jupyter Notebooks for all phases
│   ├── 01_Phase1_Exploratory_Data_Analysis.ipynb
│   ├── 02_Phase2_Recurrent_Baselines.ipynb
│   ├── 03_Phase3_Transformer_From_Scratch.ipynb
│   ├── 04_Phase4_Class_Imbalance_Solutions.ipynb
│   ├── 05_Phase5_Sequence_Readout_and_Direction.ipynb
│   ├── 06_Phase6_Pretrained_GloVe_Embeddings.ipynb
│   ├── 07_Phase7_Attention_Analysis.ipynb
│   ├── 08_Phase8_Benchmarking_and_Robustness.ipynb
│   ├── CS4045_Assignment3_Master.ipynb     # Master coordination notebook
│   └── generate_notebooks.py
├── src/                                    # Clean modular Python codebase
│   ├── data/
│   │   ├── dataset.py                      # GoEmotions Dataset, Vocab & Loader
│   │   └── eda_phase1.py                   # Filtering & EDA script
│   ├── models/
│   │   ├── rnn.py                          # Simple RNN classifier
│   │   ├── lstm.py                         # LSTM (Forward/Bi, Last/Avg/Max)
│   │   ├── gru.py                          # GRU classifier
│   │   ├── transformer.py                  # Custom Scratch Transformer Encoder
│   │   └── embeddings.py                   # GloVe download & coverage alignment
│   ├── training/
│   │   ├── losses.py                       # Weighted CE & Focal Loss
│   │   └── trainer.py                      # Training loop & early stopping
│   ├── evaluation/
│   │   ├── metrics.py                      # Accuracy, F1, McNemar test, CM plot
│   │   ├── attention_analysis.py           # Attention heatmaps & ablation
│   │   └── robustness.py                   # Test-time noise generator & curves
│   ├── utils/
│   │   ├── reproducibility.py              # Fixed random seeds (Seed=42)
│   │   ├── profiler.py                     # Parameter count & latency profiler
│   │   └── mlops_tracker.py                # MLflow & JSON experiment tracking
│   ├── run_phase2_baselines.py
│   ├── run_phase3_transformer.py
│   ├── run_phase4_imbalance.py
│   ├── run_phase5_readout.py
│   ├── run_phase6_embeddings.py
│   ├── run_phase7_attention.py
│   └── run_phase8_robustness.py
├── experiments/
│   ├── figures/                            # High-resolution plots for report
│   ├── metrics/                            # Metric JSON logs for all phases
│   └── checkpoints/                        # Saved model weights (.pt)
├── report/
│   ├── report.tex                          # Overleaf/LaTeX source code
│   └── build_report_pdf.py                 # Automated publication-grade PDF generator
├── report.pdf                              # Formal Research Report (PDF)
├── README.txt                              # This instructions file
└── .gitignore

3. ENVIRONMENT & DEPENDENCY SETUP
--------------------------------------------------------------------------------
Requirements:
- Python 3.10+
- PyTorch >= 2.0
- datasets (Hugging Face) >= 2.14
- scikit-learn >= 1.2
- pandas, numpy, scipy
- matplotlib, seaborn
- mlflow >= 2.8
- fpdf2 (for report generation)

To install all dependencies:
    pip install torch datasets scikit-learn pandas numpy scipy matplotlib seaborn mlflow fpdf2

4. HOW TO RUN THE CODE
--------------------------------------------------------------------------------
OPTION A: Running Jupyter Notebooks
Navigate to the `notebooks/` directory and launch Jupyter Lab or Notebook:
    jupyter lab
Open and execute any phase notebook sequentially (from 01 to 08), or open 
`CS4045_Assignment3_Master.ipynb` to execute the full pipeline end-to-end.

OPTION B: Running from Command Line / Terminal
You can execute each phase script directly from the project root:
    # Phase 1: Preprocessing & Exploratory Data Analysis
    python src/data/eda_phase1.py

    # Phase 2: Recurrent Baselines (Simple RNN, LSTM, GRU)
    python src/run_phase2_baselines.py

    # Phase 3: Transformer Encoder from Scratch
    python src/run_phase3_transformer.py

    # Phase 4: Class Imbalance Solutions & Retraining Canonical Models
    python src/run_phase4_imbalance.py

    # Phase 5: Sequence Readout and Direction (LSTM Study)
    python src/run_phase5_readout.py

    # Phase 6: Word Representation (Random vs Pretrained GloVe)
    python src/run_phase6_embeddings.py

    # Phase 7: Transformer Attention Interpretability & Ablation Study
    python src/run_phase7_attention.py

    # Phase 8: Benchmarking, McNemar Significance Test & Noise Robustness
    python src/run_phase8_robustness.py

OPTION C: Running on Google Colab or Kaggle
1. Upload the project folder to Google Drive or Kaggle working directory.
2. Ensure GPU runtime is enabled (or standard CPU runtime).
3. Open `notebooks/CS4045_Assignment3_Master.ipynb` and run all cells.

5. MLOPS & EXPERIMENT TRACKING (MLFLOW)
--------------------------------------------------------------------------------
All experiments automatically log parameters, metrics (train loss, val F1, test 
accuracy, macro-F1, inference latency), and model checkpoints via MLflow.
To launch the interactive MLflow tracking dashboard:
    mlflow ui
Then open http://localhost:5000 in your web browser to visually compare runs, 
metrics curves, and artifact files.

6. REPRODUCIBILITY GUARANTEE
--------------------------------------------------------------------------------
- Random seed is pinned to SEED = 42 across Python `random`, `numpy`, and `torch`
  (including CUDA deterministic mode).
- Training and validation splits are strictly honored; zero data leakage occurs.
- Model checkpoints are automatically saved to `experiments/checkpoints/`.
- Metric summaries are stored as structured JSON in `experiments/metrics/`.
- All plots are exported to `experiments/figures/` at 300 DPI.

================================================================================
