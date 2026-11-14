# Emotion Classification Benchmark on GoEmotions

An empirical evaluation comparing Recurrent Neural Networks (RNN, LSTM, GRU) and a custom scratch-built Transformer Encoder on fine-grained emotion classification across 28 categories using the GoEmotions dataset.

---

## Key Highlights

- **Strict Data Integrity**: Vocabulary (12,428 tokens) constructed strictly on the training partition with zero data leakage.
- **Custom Transformer Encoder**: Pure PyTorch implementation without high-level attention abstractions (`nn.MultiheadAttention`) or pretrained models. Features sinusoidal positional encodings, scaled dot-product attention, Pre-LN residuals, and position-wise feed-forward networks.
- **Class Imbalance Mitigation**: Solved 328:1 class skew using Class-Weighted Cross-Entropy, achieving over 4x recall improvement on minority emotion categories.
- **Sequence Readout Study**: Benchmarked 6 combinations of readout pooling (Last, Average, Max) and direction (Forward vs Bidirectional) on LSTM.
- **Pretrained GloVe Transfer**: Aligned GloVe 100d vectors (94.45% coverage), demonstrating substantial convergence and performance boosts.
- **Attention Interpretability**: Extracted attention heatmaps and conducted a controlled token ablation experiment ($k=1, 2, 3$) proving attention fidelity.
- **Benchmarking & Robustness**: Measured inference latency, model parameters, McNemar statistical significance ($p < 0.001$), and resilience against test-time typographical corruptions.

---

## Experimental Results

### Architecture Comparison (Test Set)

| Model | Test Accuracy | Macro-F1 | Parameters | Latency (Batch) |
|---|:---:|:---:|:---:|:---:|
| **Simple RNN** | 34.99% | 0.0185 | 1.28M | 7.27 ms |
| **LSTM** | **52.22%** | **0.3288** | 1.36M | 8.80 ms |
| **GRU** | 51.98% | 0.2488 | 1.33M | 23.03 ms |
| **Transformer (Scratch)** | 48.21% | 0.2649 | 1.43M | 28.18 ms |

*McNemar's test between best recurrent model and Transformer: $\chi^2 = 50.2838$, $p = 1.33 \times 10^{-12} < 0.001$.*

### Class Imbalance Mitigation (LSTM)

| Strategy | Accuracy | Macro-F1 | Macro-Recall |
|---|:---:|:---:|:---:|
| **Standard Cross-Entropy** | 53.77% | 0.2570 | 0.2511 |
| **Focal Loss ($\gamma=2$)** | 54.25% | 0.2764 | 0.2728 |
| **Class-Weighted Cross-Entropy** | 52.22% | **0.3288** | **0.3310** |

---

## Repository Structure

```
├── notebooks/                              # Jupyter Notebooks (Phases 1 to 8)
├── src/                                    # Python implementation
│   ├── data/                               # Dataset preprocessing & vocab
│   ├── models/                             # RNN, LSTM, GRU, Transformer, GloVe
│   ├── training/                           # Losses & training pipeline
│   ├── evaluation/                         # Metrics, attention, robustness
│   └── utils/                              # Reproducibility & profiler
├── experiments/
│   ├── figures/                            # Analytical charts & confusion matrices
│   ├── metrics/                            # Metric summaries (JSON)
│   └── checkpoints/                        # Trained model checkpoints (.pt)
├── report/
│   ├── report.tex                          # Formal LaTeX paper
│   └── build_report_pdf.py                 # Automated PDF builder
├── report.pdf                              # Formal Research Report (PDF)
└── README.txt                              # Standard text reference
```

---

## How to Run

Install dependencies:
```bash
pip install torch datasets scikit-learn pandas numpy scipy matplotlib seaborn mlflow fpdf2
```

Run any experimental phase directly:
```bash
# Phase 1: Preprocessing & EDA
python src/data/eda_phase1.py

# Phase 2: Recurrent Baselines
python src/run_phase2_baselines.py

# Phase 3: Transformer Encoder from Scratch
python src/run_phase3_transformer.py

# Phase 4: Class Imbalance Mitigation
python src/run_phase4_imbalance.py

# Phase 5: Sequence Readout & Direction
python src/run_phase5_readout.py

# Phase 6: Word Representation (GloVe)
python src/run_phase6_embeddings.py

# Phase 7: Attention Interpretability & Token Ablation
python src/run_phase7_attention.py

# Phase 8: Benchmarking, Statistical Test & Noise Robustness
python src/run_phase8_robustness.py
```

---

## License
MIT License
