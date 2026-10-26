import argparse
import math
import pandas as pd
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.distributed.nn.functional import all_gather as differentiable_all_gather
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data import BucketSampler, ProteinDataset, move
from .evaluation import extract, family_probes, summarize_binary
from .memory import StructuralMemory
from .models import Student
from .settings import BASE, CONFIG, STUDENT_BASE, LOCAL_ONLY, TEACHER_RUN, binary_metrics, seed, setup, write_json, configure_arm, record_initialization


class PrefixCache:
    def __init__(self, flat, lengths, initial_features, device):
        self.flat = flat
        self.lengths = lengths
        self.offsets = torch.as_tensor(np.concatenate([[0], np.cumsum(lengths)]), device=device)
        self.initial_features = initial_features

    def batch(self, indices):
        ids = torch.as_tensor(indices, device=self.flat.device)
        width = math.ceil(int(self.lengths[indices].max()) / 8) * 8
        positions = torch.arange(width, device=self.flat.device)[None, :]
        lengths = torch.as_tensor(self.lengths[indices], device=self.flat.device)[:, None]
        valid = positions < lengths
        offsets = self.offsets[ids, None] + positions
        prefix = self.flat[offsets.clamp_max(len(self.flat) - 1)] * valid.unsqueeze(-1)
        residues = valid & (positions > 0) & (positions < lengths - 1)
        return dict(prefix=prefix, attention_mask=valid.long(), residue_mask=residues)


@torch.no_grad()
def cache_prefix(model, dataset, device, rank, world):
    lengths = np.asarray([len(row) for row in dataset.encoding['input_ids']], dtype=np.int64)
    assert np.array_equal(lengths, np.asarray([len(sequence) + 2 for sequence in dataset.sequences]))
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    flat = torch.zeros((int(offsets[-1]), 320), dtype=torch.float32, device=device)
    initial = torch.zeros((len(dataset), 1280), dtype=torch.float32, device=device)
    captured = []
    hook = model.student.encoder.layer[-1].register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach()))
    indices = sorted(range(rank, len(dataset), world), key=lambda index: -lengths[index])
    loader = DataLoader(dataset, batch_size=32, sampler=indices, collate_fn=dataset.collate,
        pin_memory=True, num_workers=0)
    model.eval()
    for step, batch in enumerate(loader):
        row_ids = batch['row_ids'].numpy()
        batch = move(batch, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, features = model(batch['input_ids'], batch['attention_mask'], batch['residue_mask'])
        prefix = captured.pop()
        for position, index in enumerate(row_ids):
            flat[offsets[index]:offsets[index + 1]].copy_(prefix[position, :lengths[index]].float())
        initial[torch.as_tensor(row_ids, device=device)] = features.float()
        if step % 100 == 0:
            print('cache frozen five blocks', rank, step, flush=True)
    hook.remove()
    # Rows were computed once across the two cards; exchange the disjoint shards.
    dist.all_reduce(flat)
    dist.all_reduce(initial)
    return PrefixCache(flat, lengths, initial, device)


@torch.no_grad()
def validate(model, cache, dataset, device, rank, world):
    model.eval()
    records = []
    indices = list(range(rank, len(dataset), world))
    for start in range(0, len(indices), 16):
        rows = np.asarray(indices[start:start + 16])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(**cache.batch(rows))
        ce = F.cross_entropy(logits.float(), torch.as_tensor(dataset.labels[rows], device=device), reduction='none')
        for row, probability, loss in zip(rows, logits.float().softmax(-1)[:, 1].tolist(), ce.tolist()):
            records.append((int(row), probability, loss))
    parts = [None] * world
    dist.all_gather_object(parts, records)
    ordered = sorted(row for part in parts for row in part)
    assert [row[0] for row in ordered] == list(range(len(dataset)))
    result = binary_metrics(dataset.labels, [row[1] for row in ordered])
    result['CE'] = float(np.mean([row[2] for row in ordered]))
    return result


def main(arm):
    configure_arm(arm)
    rank, world, device = setup()
    record_initialization()
    name = arm + '_student'
    out = BASE / 'runs' / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'metrics.json').exists():
        dist.destroy_process_group()
        return
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_BASE, local_files_only=LOCAL_ONLY)
    model = Student().to(device)
    # A matched full-length, residue-pooling reference; no extra model training.
    for split in ('val', 'test', 'external', 'pace', 'family'):
        if (BASE/'data'/f'{split}.csv').exists():
            extract(model, tokenizer, 'initial_student', split, device, rank, world, batch_size=32)
    if rank == 0 and not (BASE/'evaluation/initial_student/binary_metrics.json').exists():
        summarize_binary('initial_student')
        family_probes('initial_student')
    dist.barrier()
    train_data, val_data = [ProteinDataset(split, tokenizer) for split in ('train', 'val')]
    train_cache = cache_prefix(model, train_data, device, rank, world)
    val_cache = cache_prefix(model, val_data, device, rank, world)
    archive = np.load(BASE / 'features/lodep_650m_teacher/train.npz')
    teacher_logits = torch.as_tensor(archive['logits'], device=device)
    teacher_features = torch.as_tensor(archive['features'], device=device)
    memory = None
    if arm == 'spmd':
        groups = np.load(BASE / 'structural_groups.npz')
        memory = StructuralMemory(teacher_features, train_cache.initial_features,
            torch.as_tensor(groups['assignments'],device=device),
            torch.as_tensor(groups['quality'],device=device)).to(device)
    model.train_last_block()
    optimizer = torch.optim.AdamW([
        {'params': model.student.encoder.layer[-1].parameters(), 'lr': CONFIG['student_block_lr']},
        {'params': list(model.mapping.parameters()) + list(model.classifier.parameters()),
         'lr': CONFIG['student_head_lr']}], weight_decay=CONFIG['weight_decay'])
    ddp = DDP(model, device_ids=[device.index], broadcast_buffers=False,
        find_unused_parameters=False, gradient_as_bucket_view=True)
    seed()
    sampler = BucketSampler(train_data, num_replicas=world, rank=rank, seed=42, shuffle=True)
    best_key, best_selection, history = None, None, []
    stopping_best, stale = -float('inf'), 0
    labels = torch.as_tensor(train_data.labels, device=device)
    trainables = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if rank == 0:
        write_json(out / 'config.json', {**CONFIG, 'trainable_parameters': sum(p.numel() for p in trainables),
            'total_parameters': sum(p.numel() for p in model.parameters()), 'job_id': os.environ.get('SLURM_JOB_ID'),
            'teacher_checkpoint': str(TEACHER_RUN), 'arm': arm})
        if memory is not None:
            torch.save({'teacher_prototypes': memory.teacher_prototypes.cpu(),
                'teacher_geometry': memory.teacher_geometry.cpu(), 'assignments': memory.assignments.cpu()},
                out / 'structural_memory.pt')
    for epoch in range(1, CONFIG['max_epochs'] + 1):
        sampler.set_epoch(epoch)
        indices = list(iter(sampler))
        ddp.train()
        running = torch.zeros(7, dtype=torch.float64, device=device)
        if rank == 0:
            write_json(out / 'STATUS.json', {'state': 'training', 'epoch': epoch})
        for start in range(0, len(indices), CONFIG['student_batch_per_gpu']):
            rows = np.asarray(indices[start:start + CONFIG['student_batch_per_gpu']])
            row_ids = torch.as_tensor(rows, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, features = ddp(**train_cache.batch(rows))
            temperature = CONFIG['kd_temperature']
            kd = F.kl_div((logits.float() / temperature).log_softmax(-1),
                (teacher_logits[row_ids] / temperature).softmax(-1), reduction='batchmean') * temperature**2
            ce = F.cross_entropy(logits.float(), labels[row_ids])
            feature_mse = F.mse_loss(features.float(),teacher_features[row_ids])
            query,geometry = (logits.new_zeros(()) for _ in range(2))
            if arm == 'spmd':
                global_features = torch.cat(differentiable_all_gather(features.float()))
                id_parts = [torch.empty_like(row_ids) for _ in range(world)]
                dist.all_gather(id_parts,row_ids)
                query,geometry,updated = memory(global_features,torch.cat(id_parts))
            loss = (CONFIG['kd_weight']*kd + CONFIG['classification_weight']*ce +
                CONFIG['feature_mse_weight']*feature_mse + CONFIG['prototype_query_weight']*query +
                CONFIG['prototype_geometry_weight']*geometry)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite SPMD loss')
            loss.backward()
            nn.utils.clip_grad_norm_(trainables, 1.0)
            optimizer.step()
            if memory is not None: memory.commit(updated)
            size = len(rows)
            running += torch.stack([loss.detach(), ce.detach(), kd.detach(), query.detach(),
                geometry.detach(),feature_mse.detach(),loss.detach().new_tensor(1)]).double() * size
        dist.all_reduce(running)
        validation = validate(model, val_cache, val_data, device, rank, world)
        key = validation['MCC'], validation['F1'], -validation['CE']
        if best_key is None or key > best_key:
            best_key = key
            best_selection = {'epoch': epoch, 'validation': validation}
            if rank == 0:
                state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
                temporary = out / 'student_best.pt.tmp'
                torch.save({'state_dict': state, 'student_hidden_size': 320,
                    'student_feature_dim': 1280, 'teacher_feature_dim': 1280, 'teacher_feat_dim':1280,
                    'num_labels': 2, 'method': arm, 'seed': 42, **best_selection}, temporary)
                temporary.replace(out / 'student_best.pt')
        if validation['MCC'] > stopping_best + CONFIG['min_delta']:
            stopping_best, stale = validation['MCC'], 0
        else:
            stale += 1
        record = {'epoch': epoch, 'validation': validation, 'early_stopping_wait': stale,
            'elapsed_seconds': time.time() - started,
            'train': dict(zip(['total','CE','KD','prototype_KL','prototype_geometry_MSE','feature_MSE'],
                             (running[:6] / running[6]).tolist()))}
        history.append(record)
        if rank == 0:
            write_json(out / 'history.json', history)
            print(arm, record, flush=True)
            # Epoch-boundary state allows later continuation without losing optimizer/memory state.
            torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'memory': memory.state_dict() if memory is not None else None, 'epoch': epoch, 'stopping_best': stopping_best,
                'stale_epochs': stale, 'best_selection': best_selection, 'history': history}, out / 'latest_training.pt')
        dist.barrier()
        if stale >= CONFIG['patience']:
            break
    checkpoint = torch.load(out / 'student_best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    del train_cache,val_cache,memory,teacher_logits,teacher_features,ddp,optimizer
    torch.cuda.empty_cache()
    for split in ('val', 'test', 'external', 'pace', 'family'):
        if (BASE/'data'/f'{split}.csv').exists():
            extract(model, tokenizer, name, split, device, rank, world, batch_size=32)
    if rank == 0:
        scores = summarize_binary(name)
        family = family_probes(name)
        write_json(out / 'metrics.json', {'seed': 42, 'selected_epoch': best_selection['epoch'],
            'completed_epochs': len(history), 'validation': best_selection['validation'],
            'binary': scores, 'family': family, 'seconds': time.time() - started})
        write_json(out / 'STATUS.json', {'state': 'complete', 'selected_epoch': best_selection['epoch']})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('arm',choices=['spmd','kd_output','kd_feature'])
    main(parser.parse_args().arm)
