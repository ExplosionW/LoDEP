import torch
from torch import nn
from transformers import AutoModel, AutoModelForSequenceClassification
from peft import PeftModel

from .settings import STUDENT_BASE, STUDENT_CHECKPOINT, TEACHER_BASE, TEACHER_RUN, LOCAL_ONLY


def pool(hidden, residue_mask):
    weights = residue_mask.unsqueeze(-1).float()
    return (hidden.float() * weights).sum(1) / weights.sum(1).clamp_min(1)


class OriginalTeacher(nn.Module):
    """Original LoDEP classifier and LoRA; its native CLS classifier is retained."""
    def __init__(self):
        super().__init__()
        base = AutoModelForSequenceClassification.from_pretrained(TEACHER_BASE,
            local_files_only=LOCAL_ONLY, num_labels=2, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', low_cpu_mem_usage=True)
        self.teacher = PeftModel.from_pretrained(base, TEACHER_RUN, is_trainable=False)
        self.requires_grad_(False)

    def forward(self, input_ids, attention_mask, residue_mask):
        base = self.teacher.get_base_model()
        hidden = base.esm(input_ids=input_ids, attention_mask=attention_mask,
            return_dict=True).last_hidden_state
        return base.classifier(hidden).float(), pool(hidden, residue_mask)


class Student(nn.Module):
    def __init__(self, student_base=STUDENT_BASE, checkpoint=STUDENT_CHECKPOINT):
        super().__init__()
        self.student = AutoModel.from_pretrained(student_base, local_files_only=LOCAL_ONLY,
            attn_implementation='sdpa')
        self.mapping = nn.Linear(320,1280)
        self.classifier = nn.Linear(1280,2)
        self.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict'])

    def forward(self,input_ids=None,attention_mask=None,residue_mask=None,prefix=None):
        if prefix is None:
            hidden = self.student(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        else:
            mask = (1-attention_mask[:,None,None,:].float())*torch.finfo(torch.float32).min
            hidden = self.student.encoder.layer[-1](prefix,attention_mask=mask)
            if isinstance(hidden,tuple): hidden=hidden[0]
            hidden = self.student.encoder.emb_layer_norm_after(hidden)
        features = self.mapping(pool(hidden,residue_mask))
        return self.classifier(features).float(),features.float()

    def train_last_block(self):
        self.requires_grad_(False)
        self.student.encoder.layer[-1].requires_grad_(True)
        self.mapping.requires_grad_(True)
        self.classifier.requires_grad_(True)
