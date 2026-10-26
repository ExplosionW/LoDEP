"""Train-only 3Di groups define corresponding teacher and student prototypes."""
import torch
from torch import nn
from torch.nn import functional as F

from .settings import CONFIG


def weighted_centroids(features, assignments, quality, count=64):
    valid = assignments >= 0
    sums = features.new_zeros((count, features.shape[1]))
    mass = features.new_zeros(count)
    sums.index_add_(0, assignments[valid], features[valid] * quality[valid, None])
    mass.index_add_(0, assignments[valid], quality[valid])
    return sums / mass.clamp_min(1e-8)[:, None], mass


class StructuralMemory(nn.Module):
    def __init__(self, teacher_features, initial_student_features, assignments, quality):
        super().__init__()
        count = int(assignments.max()) + 1
        if count < 1: raise ValueError("No usable structural prototypes")
        teacher, mass = weighted_centroids(teacher_features.detach().float(), assignments, quality, count)
        student, _ = weighted_centroids(initial_student_features.detach().float(), assignments, quality, count)
        if not (mass > 0).all():
            raise ValueError('Every structural prototype must have training members')
        center = teacher.mean(0, keepdim=True)
        teacher_prototypes = F.normalize(teacher - center, dim=-1)
        teacher_query = F.normalize(teacher_features.detach().float() - center, dim=-1)
        self.register_buffer('teacher_probability', (teacher_query @ teacher_prototypes.T /
            CONFIG['prototype_temperature']).softmax(-1))
        self.register_buffer('teacher_geometry', teacher_prototypes @ teacher_prototypes.T)
        self.register_buffer('teacher_prototypes', teacher_prototypes)
        self.register_buffer('student_bank', student)
        self.register_buffer('assignments', assignments)
        self.register_buffer('quality', quality)

    def forward(self, features, row_ids):
        assignments, quality = self.assignments[row_ids], self.quality[row_ids]
        means, mass = weighted_centroids(features.float(), assignments, quality, len(self.student_bank))
        touched = torch.nonzero(mass > 0, as_tuple=False).flatten()
        updated = self.student_bank.detach().clone()
        alpha = CONFIG['prototype_ema_update']
        updated = updated.index_copy(0, touched,
            (1 - alpha) * updated[touched] + alpha * means[touched])
        center = updated.mean(0, keepdim=True)
        prototypes = F.normalize(updated - center, dim=-1)
        query = F.normalize(features.float() - center, dim=-1)
        log_probability = (query @ prototypes.T / CONFIG['prototype_temperature']).log_softmax(-1)
        per_row = F.kl_div(log_probability, self.teacher_probability[row_ids], reduction='none').sum(-1)
        query_loss = (per_row * quality).sum() / quality.sum().clamp_min(1e-8)
        active = (mass > 0)[:, None] | (mass > 0)[None, :]
        active = active & ~torch.eye(len(mass), dtype=torch.bool, device=mass.device)
        differences = (prototypes @ prototypes.T - self.teacher_geometry).square()
        pair_loss = (differences * active).sum() / active.sum().clamp_min(1)
        return query_loss, pair_loss, updated

    @torch.no_grad()
    def commit(self, updated):
        self.student_bank.copy_(updated.detach())
