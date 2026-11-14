"""
Pretrained Word Embedding Utilities (GloVe 100d).
Downloads/parses GloVe embeddings, computes vocabulary coverage,
and builds initialized embedding weights for frozen/fine-tuned setups.
"""
import os
import urllib.request
import numpy as np
import torch
import torch.nn as nn
from src.data.dataset import Vocabulary

GLOVE_URL = "https://huggingface.co/arjahojnik/LSTM-sentiment-model/resolve/main/glove.6B.100d.txt"
GLOVE_CACHE_PATH = "data/glove.6B.100d.txt"

def ensure_glove_downloaded(cache_path: str = GLOVE_CACHE_PATH) -> str:
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 100_000_000:
        print(f"GloVe file found at: {cache_path}")
        return cache_path

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"Downloading GloVe 100d from Hugging Face mirror: {GLOVE_URL}...")
    
    req = urllib.request.Request(GLOVE_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp, open(cache_path, 'wb') as out_file:
        total_size = int(resp.headers.get('Content-Length', 0))
        downloaded = 0
        chunk_size = 1024 * 1024 # 1 MB
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            downloaded += len(chunk)
            out_file.write(chunk)
            if total_size > 0 and downloaded % (10 * 1024 * 1024) < chunk_size:
                print(f"  Downloaded {downloaded / (1024 * 1024):.1f} / {total_size / (1024 * 1024):.1f} MB ({downloaded/total_size*100:.1f}%)")

    print(f"GloVe download complete! Saved to {cache_path}")
    return cache_path

def build_glove_matrix(vocab: Vocabulary, glove_path: str = GLOVE_CACHE_PATH, embedding_dim: int = 100):
    ensure_glove_downloaded(glove_path)
    
    print("Reading GloVe embeddings and matching against vocabulary...")
    vocab_words = set(vocab.w2i.keys())
    matched_vectors = {}
    
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in vocab_words:
                vec = np.array(parts[1:], dtype=np.float32)
                if len(vec) == embedding_dim:
                    matched_vectors[word] = vec

    vocab_size = len(vocab)
    weights_matrix = np.random.normal(scale=0.1, size=(vocab_size, embedding_dim)).astype(np.float32)
    # Set <pad> to zeros
    weights_matrix[vocab.pad_idx] = np.zeros(embedding_dim, dtype=np.float32)

    matched_count = 0
    missing_words = []
    for word, idx in vocab.w2i.items():
        if word in matched_vectors:
            weights_matrix[idx] = matched_vectors[word]
            matched_count += 1
        elif idx not in [vocab.pad_idx, vocab.unk_idx]:
            missing_words.append(word)

    coverage_pct = (matched_count / (vocab_size - 1)) * 100 # excluding <pad>
    print(f"GloVe 100d Coverage: {matched_count}/{vocab_size - 1} vocabulary words ({coverage_pct:.2f}%)")
    print(f"Sample missing words (e.g. Reddit internet slang/contractions): {missing_words[:10]}")

    return torch.tensor(weights_matrix, dtype=torch.float), {
        'total_vocab': vocab_size,
        'matched_count': matched_count,
        'coverage_pct': round(coverage_pct, 2),
        'missing_count': len(missing_words),
        'sample_missing': missing_words[:15]
    }
