#!/usr/bin/env python3
"""
Enhanced train_LORA.py
- Cleaner teacher LoRA training (using HuggingFace Trainer + PEFT)
- Prep teacher outputs (logits + pooled features -> .npy)
- Student distillation training with stable saving of state + metadata
- Saves student checkpoint as a dictionary: { 'state_dict': ..., 'teacher_feat_dim': ..., 'num_labels': ... }
- Added evaluation hooks and optional freezing of student base

Usage examples (short):
# Train teacher (LoRA)
python train_LORA.py --stage teacher --train_csv dataset/train.csv --val_csv dataset/validation.csv \
    --tokenizer facebook/esm2_t33_650M_UR50D --teacher_model facebook/esm2_t33_650M_UR50D \
    --output_dir ./teacher_out --num_train_epochs 3 --per_device_train_batch_size 16

# Export teacher logits/features
python train_LORA.py --stage prep_teacher_outputs --train_csv dataset/train.csv \
    --tokenizer facebook/esm2_t33_650M_UR50D --teacher_model facebook/esm2_t33_650M_UR50D \
    --teacher_dir ./teacher_out --teacher_logits_out teacher_logits.npy --teacher_feats_out teacher_feats.npy

# Train student
python train_LORA.py --stage student --train_csv dataset/train.csv --val_csv dataset/validation.csv \
    --tokenizer facebook/esm2_t6_8M_UR50D --student_model facebook/esm2_t6_8M_UR50D \
    --teacher_logits teacher_logits.npy --teacher_features teacher_feats.npy --output_dir_student ./student_out --epochs 10

"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoModel,
    Trainer, TrainingArguments, default_data_collator, TrainerCallback
)
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class VRAMUsageCallback(TrainerCallback):
    """
    [VRAM Monitor] 小模块：在每个 epoch 结束后，打印当前 GPU 显存的使用情况 (MB 和 GB)
    """
    def on_epoch_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
            max_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
            print(f"\n[Epoch {state.epoch:.2f} VRAM Usage]")
            print(f"  - 已分配 (Allocated): {allocated_mb:.2f} MB ({allocated_mb/1024:.2f} GB)")
            print(f"  - 已保留 (Reserved) : {reserved_mb:.2f} MB ({reserved_mb/1024:.2f} GB)")
            print(f"  - 峰值分配 (Peak)   : {max_allocated_mb:.2f} MB ({max_allocated_mb/1024:.2f} GB)\n")
            torch.cuda.reset_peak_memory_stats()


# -------------------------
# Loss helpers
# -------------------------

def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=2.0):
    # student_logits, teacher_logits: (B, C)
    ce = F.cross_entropy(student_logits, labels)
    T = temperature
    p_student = F.log_softmax(student_logits / T, dim=1)
    p_teacher = F.softmax(teacher_logits / T, dim=1)
    kld = F.kl_div(p_student, p_teacher, reduction='batchmean') * (T * T)
    return alpha * kld + (1.0 - alpha) * ce


def feature_alignment_loss(student_feats, teacher_feats):
    return F.mse_loss(student_feats, teacher_feats)


# -------------------------
# Data utilities
# -------------------------

def read_csv_to_dataset(train_csv, val_csv=None, seq_col="seq", label_col="label"):
    train_df = pd.read_csv(train_csv)
    if val_csv is not None:
        val_df = pd.read_csv(val_csv)
    else:
        val_df = None
    ds = {"train": Dataset.from_pandas(train_df)}
    if val_df is not None:
        ds["validation"] = Dataset.from_pandas(val_df)
    return DatasetDict(ds)


def resolve_max_length(tokenizer, sequences, requested_max_length=None):
    if requested_max_length is not None:
        return requested_max_length

    sequence_list = [str(sequence) for sequence in sequences]
    max_token_length = 0
    for start in range(0, len(sequence_list), 1000):
        batch = sequence_list[start:start + 1000]
        encoded = tokenizer(batch, add_special_tokens=True, padding=False, truncation=False)
        max_token_length = max(max_token_length, max(len(ids) for ids in encoded["input_ids"]))

    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 100000:
        max_token_length = min(max_token_length, tokenizer_limit)

    print(f"[data] using max_length={max_token_length} (auto)")
    return max_token_length


def preprocess_tokenize(tokenizer, dataset_dict, max_length=None, seq_col="seq"):
    all_sequences = []
    for split in dataset_dict.values():
        all_sequences.extend(split[seq_col])
    max_length = resolve_max_length(tokenizer, all_sequences, max_length)

    def preprocess_function(examples):
        sequences = [str(s) for s in examples[seq_col]]
        return tokenizer(sequences, padding="max_length", truncation=True, max_length=max_length)
    tokenized = dataset_dict.map(preprocess_function, batched=True)
    # ensure label column name
    if "label" in tokenized["train"].column_names:
        tokenized = tokenized.rename_column("label", "labels")
    return tokenized


# -------------------------
# Teacher training with LoRA
# -------------------------

def train_teacher(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[teacher] device = {device}")

    ds = read_csv_to_dataset(args.train_csv, args.val_csv)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenized = preprocess_tokenize(tokenizer, ds, max_length=args.max_length)

    num_labels = int(pd.read_csv(args.train_csv)["label"].nunique())
    print(f"[teacher] num_labels = {num_labels}")

    print("[teacher] loading base model...")
    base_model = AutoModelForSequenceClassification.from_pretrained(args.teacher_model, num_labels=num_labels)

    print("[teacher] Applying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(",") if args.lora_target_modules else ["query", "key", "value"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(base_model, lora_config)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[teacher] total params: {total:,}, trainable params: {trainable:,}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True if args.load_best else False,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        logging_steps=50,
        fp16=torch.cuda.is_available()
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if "validation" in tokenized else None,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[VRAMUsageCallback()]
    )

    print("[teacher] Starting training...")
    trainer.train()

    # Save adapter + tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    print("[teacher] Saving adapter & tokenizer...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[teacher] Done.")


# -------------------------
# Prep teacher outputs
# -------------------------

def prep_teacher_outputs(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[prep_teacher_outputs] device = {device}")

    df = pd.read_csv(args.train_csv)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    sequences = df["seq"].astype(str).tolist()
    max_length = resolve_max_length(tokenizer, sequences, args.max_length)
    enc = tokenizer(sequences, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

    base_model = AutoModelForSequenceClassification.from_pretrained(args.teacher_model, num_labels=int(df["label"].nunique()))
    print("[prep_teacher_outputs] Loading PEFT adapter...")
    peft_model = PeftModel.from_pretrained(base_model, args.teacher_dir, is_trainable=False).to(device)
    peft_model.eval()

    batch_size = args.per_device_eval_batch_size
    logits_list = []
    feats_list = []
    with torch.no_grad():
        for i in tqdm(range(0, enc["input_ids"].shape[0], batch_size), desc="teacher inference"):
            batch_ids = enc["input_ids"][i:i+batch_size].to(device)
            batch_mask = enc["attention_mask"][i:i+batch_size].to(device)
            outputs = peft_model(batch_ids, attention_mask=batch_mask, output_hidden_states=True)
            logits = outputs.logits.detach().cpu().numpy()
            last_hidden = outputs.hidden_states[-1]
            pooled = last_hidden.mean(dim=1).detach().cpu().numpy()
            logits_list.append(logits)
            feats_list.append(pooled)

    logits_arr = np.concatenate(logits_list, axis=0)
    feats_arr = np.concatenate(feats_list, axis=0)

    np.save(args.teacher_logits_out, logits_arr)
    np.save(args.teacher_feats_out, feats_arr)
    print(f"[prep_teacher_outputs] saved logits -> {args.teacher_logits_out}, feats -> {args.teacher_feats_out}")


# -------------------------
# Student model and training
# -------------------------
class StudentWithMapping(nn.Module):
    def __init__(self, student_base, student_hidden_size, teacher_feat_dim, num_labels):
        super().__init__()
        self.student = student_base
        self.mapping = nn.Linear(student_hidden_size, teacher_feat_dim)
        self.classifier = nn.Linear(teacher_feat_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hs = outputs.hidden_states[-1]
        pooled = last_hs.mean(dim=1)
        mapped = self.mapping(pooled)
        logits = self.classifier(mapped)
        return logits, mapped


def train_student(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[student] device = {device}")

    train_df = pd.read_csv(args.train_csv)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    train_sequences = train_df["seq"].astype(str).tolist()
    length_sequences = list(train_sequences)
    val_df = None
    if args.val_csv:
        val_df = pd.read_csv(args.val_csv)
        length_sequences.extend(val_df["seq"].astype(str).tolist())
    max_length = resolve_max_length(tokenizer, length_sequences, args.max_length)

    enc = tokenizer(train_sequences, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    labels = torch.tensor(train_df["label"].tolist(), dtype=torch.long)

    # load teacher outputs
    teacher_logits = np.load(args.teacher_logits)
    teacher_feats = np.load(args.teacher_features)
    assert teacher_logits.shape[0] == enc["input_ids"].shape[0], "teacher logits count mismatch"
    assert teacher_feats.shape[0] == enc["input_ids"].shape[0], "teacher feats count mismatch"

    dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], labels,
                            torch.tensor(teacher_logits, dtype=torch.float32),
                            torch.tensor(teacher_feats, dtype=torch.float32))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Load validation set if provided
    val_dataloader = None
    if val_df is not None:
        val_enc = tokenizer(val_df["seq"].astype(str).tolist(), padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        val_labels = torch.tensor(val_df["label"].tolist(), dtype=torch.long)
        val_dataset = TensorDataset(val_enc["input_ids"], val_enc["attention_mask"], val_labels)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        print(f"[student] validation set loaded: {len(val_df)} samples")

    student_base = AutoModel.from_pretrained(args.student_model)
    student_hidden_size = student_base.config.hidden_size
    teacher_feat_dim = teacher_feats.shape[1]
    num_labels = teacher_logits.shape[1]

    student_model = StudentWithMapping(student_base, student_hidden_size, teacher_feat_dim, num_labels).to(device)

    if args.freeze_student_base:
        for p in student_model.student.parameters():
            p.requires_grad = False
        print("[student] student base frozen; training only mapping+classifier")

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.student_lr)

    best_prec = 0.0
    best_ckpt = None

    for epoch in range(1, args.epochs + 1):
        student_model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            optimizer.zero_grad()
            input_ids, attn_mask, lab, t_logits, t_feats = batch
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            lab = lab.to(device)
            t_logits = t_logits.to(device)
            t_feats = t_feats.to(device)

            s_logits, s_mapped = student_model(input_ids, attn_mask)

            loss_logits = distillation_loss(s_logits, t_logits, lab, alpha=args.alpha, temperature=args.temperature)
            loss_feats = feature_alignment_loss(s_mapped, t_feats)
            loss = loss_logits + args.beta * loss_feats

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"[student] epoch {epoch} avg loss {avg_loss:.4f}")

        # eval on validation set (or fallback to training set)
        student_model.eval()
        all_preds = []
        all_labels = []
        eval_loader = val_dataloader if val_dataloader is not None else dataloader
        eval_name = "val" if val_dataloader is not None else "train"
        with torch.no_grad():
            for batch in eval_loader:
                if val_dataloader is not None:
                    input_ids, attn_mask, lab = batch
                else:
                    input_ids, attn_mask, lab, _, _ = batch
                input_ids = input_ids.to(device); attn_mask = attn_mask.to(device)
                logits, _ = student_model(input_ids, attn_mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(lab.numpy())
        from sklearn.metrics import precision_score
        prec = precision_score(all_labels, all_preds, zero_division=0)
        print(f"[student] epoch {epoch} {eval_name} precision {prec:.4f}")
        if prec > best_prec:
            best_prec = prec
            # save full checkpoint with metadata
            ckpt = {
                'state_dict': {k: v.cpu() for k, v in student_model.state_dict().items()},
                'teacher_feat_dim': teacher_feat_dim,
                'num_labels': num_labels,
                'student_hidden_size': student_hidden_size
            }
            best_ckpt = ckpt

        # [VRAM Monitor] 小模块：在每个 epoch 结束后，打印当前 GPU 显存的使用情况 (MB 和 GB)
        # if torch.cuda.is_available():
        #     max_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        #     print(f"[{'='*20}]")
        #     print(f"[VRAM Monitor] Epoch {epoch} 结束 - 最大显存消耗: {max_vram_mb:.2f} MB")
        #     print(f"[{'='*20}]\n")
        #     torch.cuda.reset_peak_memory_stats()
        if torch.cuda.is_available():
            allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
            max_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
            print(f"\n[Epoch {epoch:.2f} VRAM Usage]")
            print(f"  - 已分配 (Allocated): {allocated_mb:.2f} MB ({allocated_mb/1024:.2f} GB)")
            print(f"  - 已保留 (Reserved) : {reserved_mb:.2f} MB ({reserved_mb/1024:.2f} GB)")
            print(f"  - 峰值分配 (Peak)   : {max_allocated_mb:.2f} MB ({max_allocated_mb/1024:.2f} GB)\n")
            torch.cuda.reset_peak_memory_stats()
    os.makedirs(args.output_dir, exist_ok=True)
    if best_ckpt is not None:
        out_path = os.path.join(args.output_dir, "student_best_state.pt")
        torch.save(best_ckpt, out_path)
        # also save the base student model for compatibility (optional)
        student_model.student.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"[student] saved best checkpoint precision={best_prec:.4f} -> {out_path}")
    else:
        print("[student] no checkpoint to save")

    print(f"[student] done. best_prec={best_prec:.4f}")


# -------------------------
# arg parsing + main
# -------------------------

def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["teacher", "prep_teacher_outputs", "student"], required=True)

    # data
    p.add_argument("--train_csv", type=str)
    p.add_argument("--val_csv", type=str, default=None)
    p.add_argument("--max_length", type=int, default=None)

    # teacher config
    p.add_argument("--teacher_model", type=str, default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--tokenizer", type=str, default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--output_dir", type=str, default="./teacher_out")
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--load_best", action="store_true")

    # LoRA teacher
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--lora_target_modules", type=str, default="query,key,value")

    # prep outputs
    p.add_argument("--teacher_dir", type=str, default="./teacher_out")
    p.add_argument("--teacher_logits_out", type=str, default="teacher_logits.npy")
    p.add_argument("--teacher_feats_out", type=str, default="teacher_feats.npy")

    # student config
    p.add_argument("--student_model", type=str, default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--student_lr", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--temperature", type=float, default=5.0)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--teacher_logits", type=str, default="teacher_logits.npy")
    p.add_argument("--teacher_features", type=str, default="teacher_feats.npy")
    p.add_argument("--output_dir_student", type=str, default="./student_out")
    p.add_argument("--freeze_student_base", action="store_true")

    return p


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.stage == "teacher":
        assert args.train_csv is not None
        train_teacher(args)
    elif args.stage == "prep_teacher_outputs":
        assert args.train_csv is not None
        prep_teacher_outputs(args)
    elif args.stage == "student":
        # map names for compatibility
        args.output_dir = args.output_dir_student
        assert args.train_csv is not None and os.path.exists(args.teacher_logits) and os.path.exists(args.teacher_features)
        train_student(args)
    else:
        raise ValueError("Unknown stage")


if __name__ == "__main__":
    main()


# 微调 0.9674
# & "D:\anconda\envs\biotransformers\python.exe" `
# "D:\python\python project\DL\LORA\train_LORA.py" `
# --stage teacher `
# --train_csv "dataset/train.csv" `
# --val_csv "dataset/validation.csv" `
# --tokenizer "facebook/esm2_t33_650M_UR50D" `
# --teacher_model "facebook/esm2_t33_650M_UR50D" `
# --output_dir "D:\python\python project\DL\LORA\teacher_out" `
# --num_train_epochs 8 `
# --per_device_train_batch_size 8 `
# --per_device_eval_batch_size 8 `
# --learning_rate 2e-5 `
# --weight_decay 0.01 `
# --load_best `

# npy
# & "D:\anconda\envs\biotransformers\python.exe" `
# "D:\python\python project\DL\LORA\train_LORA.py" `
# --stage prep_teacher_outputs `
# --train_csv "dataset/train.csv" `
# --tokenizer "facebook/esm2_t33_650M_UR50D" `
# --teacher_model "facebook/esm2_t33_650M_UR50D" `
# --teacher_dir "D:\python\python project\DL\LORA\teacher_out" `
# --teacher_logits_out "D:\python\python project\DL\LORA\teacher_logits.npy" `
# --teacher_feats_out "D:\python\python project\DL\LORA\teacher_features.npy" `
# --per_device_eval_batch_size 8

# train student
# & "D:\anconda\envs\biotransformers\python.exe" `
# "D:\python\python project\DL\LORA\train_LORA.py" `
# --stage student `
# --train_csv "dataset/train.csv" `
# --tokenizer "facebook/esm2_t6_8M_UR50D" `
# --student_model "facebook/esm2_t6_8M_UR50D" `
# --teacher_logits "D:\python\python project\DL\LORA\teacher_logits.npy" `
# --teacher_features "D:\python\python project\DL\LORA\teacher_features.npy" `
# --output_dir_student "D:\python\python project\DL\LORA\student_out" `
# --epochs 10 `
# --batch_size 32 `
# --student_lr 5e-5 `
# --alpha 0.7 `
# --beta 0.3 `
# --temperature 5.0 `
