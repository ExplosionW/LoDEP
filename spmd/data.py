import math

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DistributedSampler

from .settings import BASE, CONFIG


class ProteinDataset(Dataset):
    def __init__(self, split, tokenizer, frame=None):
        self.frame = pd.read_csv(BASE / 'data' / f'{split}.csv') if frame is None else frame.copy()
        self.sequences = self.frame['sequence'].tolist()
        self.labels = self.frame['label'].to_numpy(dtype=np.int64)
        texts = self.sequences
        self.encoding = tokenizer(texts, add_special_tokens=True, padding=False,
            truncation=False, return_special_tokens_mask=True)
        self.positions = [[position for position, special in enumerate(row) if not special]
                          for row in self.encoding['special_tokens_mask']]
        if [len(row) for row in self.positions] != [len(sequence) for sequence in self.sequences]:
            raise ValueError(f'{split}: tokenizer/residue alignment mismatch')
        self.pad_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return index

    def collate(self, indices):
        width = math.ceil(max(len(self.encoding['input_ids'][index]) for index in indices) / 8) * 8
        ids = torch.full((len(indices), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        residues = torch.zeros_like(ids, dtype=torch.bool)
        for row, index in enumerate(indices):
            tokens = self.encoding['input_ids'][index]
            ids[row, :len(tokens)] = torch.tensor(tokens)
            mask[row, :len(tokens)] = 1
            residues[row, self.positions[index]] = True
        return dict(input_ids=ids, attention_mask=mask, residue_mask=residues,
            labels=torch.tensor(self.labels[indices]), row_ids=torch.tensor(indices),
            structure_y=torch.full_like(ids, -100), structure_confidence=torch.zeros_like(ids, dtype=torch.float32))


class BucketSampler(DistributedSampler):
    def __iter__(self):
        generator = torch.Generator().manual_seed(42 + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        ordered = []
        for start in range(0, len(indices), 128):
            ordered.extend(sorted(indices[start:start + 128],
                key=lambda index: -len(self.dataset.sequences[index])))
        if len(ordered) < self.total_size:
            ordered += ordered[:self.total_size - len(ordered)]
        return iter(ordered[self.rank:self.total_size:self.num_replicas])


def move(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}
