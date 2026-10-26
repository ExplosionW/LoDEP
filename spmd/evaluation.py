import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_recall_fscore_support, precision_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import ProteinDataset, move
from .settings import BASE, FAMILIES, binary_metrics, write_json


@torch.no_grad()
def extract(model, tokenizer, name, split, device, rank, world, batch_size=16):
    folder = BASE / 'features' / name
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / f'{split}.npz'
    if final.exists():
        return
    dataset = ProteinDataset(split, tokenizer)
    if len(dataset)<world: raise ValueError('Evaluation split must have at least one row per rank')
    indices = sorted(range(rank, len(dataset), world), key=lambda index: -len(dataset.sequences[index]))
    loader = DataLoader(dataset, batch_size=batch_size, sampler=indices,
        collate_fn=dataset.collate, pin_memory=True, num_workers=0)
    features, logits, ids = [], [], []
    model.eval()
    for step, batch in enumerate(loader):
        ids.append(batch['row_ids'].numpy())
        batch = move(batch, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output, feature = model(batch['input_ids'], batch['attention_mask'], batch['residue_mask'])
        if not torch.isfinite(output).all():
            raise FloatingPointError('Nonfinite inference logits')
        features.append(feature.float().cpu().numpy())
        logits.append(output.float().cpu().numpy())
        if step % 100 == 0:
            print(name, split, 'rank', rank, 'batch', step, flush=True)
    np.savez(folder / f'{split}_rank{rank}.npz', row_ids=np.concatenate(ids),
        features=np.concatenate(features), logits=np.concatenate(logits))
    dist.barrier()
    if rank == 0:
        parts = [np.load(folder / f'{split}_rank{worker}.npz') for worker in range(world)]
        row_ids = np.concatenate([part['row_ids'] for part in parts])
        order = np.argsort(row_ids)
        assert np.array_equal(row_ids[order], np.arange(len(dataset)))
        temporary = folder / f'{split}.partial.npz'
        np.savez(temporary, row_ids=row_ids[order],
            features=np.concatenate([part['features'] for part in parts])[order],
            logits=np.concatenate([part['logits'] for part in parts])[order])
        temporary.replace(final)
        for part in parts:
            part.close()
    dist.barrier()


def summarize_binary(name):
    scores = {}
    out = BASE / 'evaluation' / name
    out.mkdir(parents=True, exist_ok=True)
    for split in ('val', 'test', 'external', 'pace'):
        path=BASE / 'features' / name / f'{split}.npz'
        if not path.exists():continue
        archive = np.load(path)
        labels = pd.read_csv(BASE / 'data' / f'{split}.csv')['label'].to_numpy()
        logits = torch.from_numpy(archive['logits']).float()
        probability = logits.softmax(-1)[:, 1].numpy()
        scores[split] = binary_metrics(labels, probability)
        scores[split]['CE'] = float(F.cross_entropy(logits, torch.from_numpy(labels)))
        pd.DataFrame({'row_id': np.arange(len(labels)), 'label': labels, 'probability': probability,
            'prediction': (probability >= 0.5).astype(int)}).to_csv(out / f'{split}_predictions.csv', index=False)
    write_json(out / 'binary_metrics.json', scores)
    return scores


def family_probes(name):
    if not (BASE/'data/family.csv').exists():return {}
    frame = pd.read_csv(BASE / 'data/family.csv')
    labels = frame['family_id'].to_numpy()
    features = np.load(BASE / 'features' / name / 'family.npz')['features']
    splits = json.loads((BASE / 'data/family_splits.json').read_text())
    out = BASE / 'evaluation' / name
    out.mkdir(parents=True, exist_ok=True)
    results, per_class, predictions = {}, [], []
    for regime, split in splits.items():
        support, validation, test = [np.asarray(split[key], dtype=int) for key in ('train', 'val', 'test')]
        assert set(support).isdisjoint(test) and set(validation).isdisjoint(test)
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0,
            class_weight='balanced', max_iter=1000, solver='lbfgs', random_state=42))
        probe.fit(features[support], labels[support])
        predicted = probe.predict(features[test])
        precision, recall, f1, count = precision_recall_fscore_support(
            labels[test], predicted, labels=np.arange(6), zero_division=0)
        matrix = confusion_matrix(labels[test], predicted, labels=np.arange(6))
        results[regime] = dict(n_support=len(support), n_validation=len(validation), n_test=len(test),
            ACC=float(accuracy_score(labels[test], predicted)),
            macro_precision=float(precision.mean()), macro_recall=float(recall.mean()),
            macro_F1=float(f1.mean()), MCC=float(matthews_corrcoef(labels[test], predicted)),
            weighted_F1=float(f1_score(labels[test], predicted, average='weighted', zero_division=0)),
            PACE_precision=float(precision[3]), PACE_recall=float(recall[3]), PACE_F1=float(f1[3]),
            PACE_n_test=int(count[3]), confusion_matrix=matrix.tolist())
        for index, family in enumerate(FAMILIES):
            per_class.append(dict(regime=regime, family=family, precision=precision[index],
                recall=recall[index], F1=f1[index], n_test=int(count[index]),
                TP=int(matrix[index, index]), FP=int(matrix[:, index].sum() - matrix[index, index])))
        for row_id, prediction in zip(test, predicted):
            predictions.append(dict(regime=regime, row_id=int(row_id),
                true_family=FAMILIES[labels[row_id]], predicted_family=FAMILIES[prediction]))
        joblib.dump(probe, out / f'probe_{regime.replace("%", "pct")}.joblib')
    write_json(out / 'family_metrics.json', results)
    pd.DataFrame(per_class).to_csv(out / 'family_per_class.csv', index=False)
    pd.DataFrame(predictions).to_csv(out / 'family_predictions.csv', index=False)
    return results
