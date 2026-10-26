"""Portable paths and the saved EP650M protocol; no cluster-specific imports."""
import datetime
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)

REPO=Path(__file__).resolve().parents[1]
BASE=Path(os.environ.get('SPMD_WORK_DIR',REPO/'outputs/spmd_seed42')).resolve()
TEACHER_BASE=os.environ.get('SPMD_TEACHER_BASE','facebook/esm2_t33_650M_UR50D')
TEACHER_RUN=os.environ.get('SPMD_TEACHER_DIR',str(REPO/'LORA_teacher'))
STUDENT_BASE=os.environ.get('SPMD_STUDENT_BASE',str(REPO/'student_model'))
STUDENT_CHECKPOINT=Path(os.environ.get('SPMD_STUDENT_CHECKPOINT',REPO/'student_model/student_best_state.pt'))
LOCAL_ONLY=os.environ.get('SPMD_LOCAL_ONLY','0')=='1'
FAMILIES=['ABC','MFS','RND','PACE','MATE','SMR']
CONFIG=dict(seed=42,max_epochs=20,patience=3,min_delta=1e-4,
    monitor='validation MCC',checkpoint_ties=['F1','lower classification CE'],
    sequence_policy='full_length_no_truncation',pooling='valid residues only; special/padding tokens excluded',
    student='Original LoDEP8M checkpoint; final ESM block plus mapping/classifier trainable',
    student_batch_per_gpu=int(os.environ.get('SPMD_BATCH_SIZE','16')),
    world_size=int(os.environ.get('WORLD_SIZE','1')),
    student_block_lr=1e-5,student_head_lr=5e-5,weight_decay=.01,
    kd_temperature=5.,kd_weight=.7,classification_weight=.3,feature_mse_weight=.3,
    prototype_query_weight=.3,prototype_geometry_weight=.3,prototype_temperature=.1,
    prototype_ema_update=.1,prototypes_per_class=32,structure_min_confidence=.5,
    descriptor='sqrt-normalized confidence-weighted20-state counts and400 adjacent-state counts',
    teacher='Original650M LoDEP LoRA checkpoint',teacher_has_3di_auxiliary_training=False,
    method='SPMD classification-teacher plus predicted3Di memory variant; working name',
    prototype_fit_scope='training split only',threshold=.5)
CONFIG['effective_batch_size']=CONFIG['student_batch_per_gpu']*CONFIG['world_size']

def configure_arm(arm):
    if arm not in ('spmd','kd_output','kd_feature'):raise ValueError(arm)
    CONFIG.update(arm=arm,feature_mse_weight=0. if arm=='kd_output' else .3,
        prototype_query_weight=.3 if arm=='spmd' else 0.,
        prototype_geometry_weight=.3 if arm=='spmd' else 0.)

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)

def setup():
    world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    if world not in (1,2):raise ValueError('Use one node with one or two CUDA ranks')
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for training/export; CPU core tests use no CUDA')
    torch.cuda.set_device(local)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=90))
    seed();return dist.get_rank(),world,torch.device('cuda',local)

def binary_metrics(y,probability):
    y=np.asarray(y);probability=np.asarray(probability);predicted=(probability>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,predicted,labels=[0,1]).ravel()
    return dict(n=int(len(y)),ACC=float(accuracy_score(y,predicted)),
        F1=float(f1_score(y,predicted,zero_division=0)),macro_F1=float(f1_score(y,predicted,average='macro',zero_division=0)),
        MCC=float(matthews_corrcoef(y,predicted)),precision=float(precision_score(y,predicted,zero_division=0)),
        recall=float(recall_score(y,predicted,zero_division=0)),
        AUROC=float(roc_auc_score(y,probability)) if len(np.unique(y))==2 else None,
        AUPRC=float(average_precision_score(y,probability)) if len(np.unique(y))==2 else None,
        TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp))
