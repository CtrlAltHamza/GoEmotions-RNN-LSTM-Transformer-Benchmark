"""
Data processing and PyTorch Dataset/DataLoader pipeline for GoEmotions.
Strictly builds vocabulary on training split.
Handles single-label filtering and padding/truncation.
"""
import re
import torch
from torch.utils.data import Dataset, DataLoader
import datasets
from collections import Counter

SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1
}

def tokenize_text(text: str) -> list[str]:
    """Tokenize lowercased text into alphanumeric words and apostrophes."""
    return re.findall(r"\b\w+(?:'\w+)?\b|[^\w\s]", text.lower())

class Vocabulary:
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.w2i = dict(SPECIAL_TOKENS)
        self.i2w = {idx: word for word, idx in SPECIAL_TOKENS.items()}
        self.pad_idx = SPECIAL_TOKENS["<pad>"]
        self.unk_idx = SPECIAL_TOKENS["<unk>"]

    def build_vocab(self, texts: list[str]):
        counter = Counter()
        for text in texts:
            tokens = tokenize_text(text)
            counter.update(tokens)
        
        current_idx = len(self.w2i)
        for word, count in counter.items():
            if count >= self.min_freq and word not in self.w2i:
                self.w2i[word] = current_idx
                self.i2w[current_idx] = word
                current_idx += 1

    def encode(self, tokens: list[str], max_len: int = 50) -> list[int]:
        ids = [self.w2i.get(token, self.unk_idx) for token in tokens[:max_len]]
        if len(ids) < max_len:
            ids += [self.pad_idx] * (max_len - len(ids))
        return ids

    def decode(self, token_ids: list[int]) -> list[str]:
        return [self.i2w.get(idx, "<unk>") for idx in token_ids if idx != self.pad_idx]

    def __len__(self):
        return len(self.w2i)

class GoEmotionsDataset(Dataset):
    def __init__(self, items: list[dict], vocab: Vocabulary, max_len: int = 50):
        self.items = items
        self.vocab = vocab
        self.max_len = max_len
        self.encoded_data = []
        for item in items:
            tokens = tokenize_text(item['text'])
            encoded = vocab.encode(tokens, max_len=max_len)
            self.encoded_data.append({
                'input_ids': torch.tensor(encoded, dtype=torch.long),
                'label': torch.tensor(item['label'], dtype=torch.long),
                'length': min(len(tokens), max_len),
                'text': item['text'],
                'tokens': tokens[:max_len]
            })

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]

def get_go_emotions_data(max_len: int = 50, min_freq: int = 2, batch_size: int = 64):
    """
    Load GoEmotions simplified dataset, filter single-label comments,
    build vocabulary from train split ONLY, and return DataLoaders.
    """
    raw_ds = datasets.load_dataset('google-research-datasets/go_emotions', 'simplified')
    label_names = raw_ds['train'].features['labels'].feature.names
    
    filtered_splits = {}
    for split in ['train', 'validation', 'test']:
        single_items = []
        for item in raw_ds[split]:
            if len(item['labels']) == 1:
                single_items.append({'text': item['text'], 'label': item['labels'][0], 'id': item['id']})
        filtered_splits[split] = single_items

    # Build vocab strictly from train split
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build_vocab([item['text'] for item in filtered_splits['train']])

    train_ds = GoEmotionsDataset(filtered_splits['train'], vocab, max_len=max_len)
    val_ds = GoEmotionsDataset(filtered_splits['validation'], vocab, max_len=max_len)
    test_ds = GoEmotionsDataset(filtered_splits['test'], vocab, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_ds': train_ds,
        'val_ds': val_ds,
        'test_ds': test_ds,
        'vocab': vocab,
        'label_names': label_names,
        'num_classes': len(label_names)
    }

if __name__ == '__main__':
    data_bundle = get_go_emotions_data()
    print(f"Vocab size: {len(data_bundle['vocab'])}")
    print(f"Train batches: {len(data_bundle['train_loader'])}")
    print(f"Val batches: {len(data_bundle['val_loader'])}")
    print(f"Test batches: {len(data_bundle['test_loader'])}")
