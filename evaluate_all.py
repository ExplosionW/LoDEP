#!/usr/bin/env python3
"""Evaluate LoDEP teacher and student models on a labeled test CSV."""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


class StudentWithMapping(nn.Module):
    """LoDEP distilled student architecture saved by train_LORA.py."""

    def __init__(self, student_base, student_hidden_size, teacher_feat_dim, num_labels):
        super().__init__()
        self.student = student_base
        self.mapping = nn.Linear(student_hidden_size, teacher_feat_dim)
        self.classifier = nn.Linear(teacher_feat_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        pooled = outputs.hidden_states[-1].mean(dim=1)
        mapped = self.mapping(pooled)
        logits = self.classifier(mapped)
        return logits, mapped


def build_dataloader(test_csv, tokenizer_path, max_length, batch_size):
    df = pd.read_csv(test_csv)
    if "seq" not in df.columns or "label" not in df.columns:
        raise ValueError("Test CSV must contain 'seq' and 'label' columns.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    encodings = tokenizer(
        df["seq"].astype(str).tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = torch.tensor(df["label"].astype(int).tolist(), dtype=torch.long)
    dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], labels)

    def collate_fn(batch):
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        batch_labels = torch.stack([item[2] for item in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": batch_labels,
        }

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


def collect_predictions(model, dataloader, device, distilled=False):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if distilled:
                logits, _ = model(input_ids, attention_mask)
            else:
                logits = model(input_ids, attention_mask).logits

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
    }


def evaluate_teacher(args, device):
    print(f"\nEvaluating LoDEP teacher: {args.lora_teacher}")
    base = AutoModelForSequenceClassification.from_pretrained(
        args.lora_teacher_base,
        num_labels=args.num_labels,
    )
    model = PeftModel.from_pretrained(base, args.lora_teacher).to(device)
    dataloader = build_dataloader(
        args.test_csv,
        args.lora_teacher_base,
        args.max_length,
        args.batch_size,
    )
    y_true, y_pred, y_prob = collect_predictions(model, dataloader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["Model"] = "LoDEP_teacher"
    return metrics


def evaluate_student(args, device):
    print(f"\nEvaluating LoDEP student: {args.lora_student}")
    checkpoint = torch.load(args.lora_student, map_location="cpu")
    base = AutoModel.from_pretrained(args.lora_student_base)
    model = StudentWithMapping(
        base,
        student_hidden_size=checkpoint["student_hidden_size"],
        teacher_feat_dim=checkpoint["teacher_feat_dim"],
        num_labels=checkpoint["num_labels"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    dataloader = build_dataloader(
        args.test_csv,
        args.lora_student_base,
        args.max_length,
        args.batch_size,
    )
    y_true, y_pred, y_prob = collect_predictions(
        model,
        dataloader,
        device,
        distilled=True,
    )
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["Model"] = "LoDEP_student"
    return metrics


def main():
    parser = argparse.ArgumentParser("Evaluate LoDEP teacher and student models.")
    parser.add_argument("--test_csv", type=str, default="dataset/test.csv")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--output_csv", type=str, default="evaluation_results.csv")

    parser.add_argument("--lora_teacher", type=str, default=None)
    parser.add_argument(
        "--lora_teacher_base",
        type=str,
        default="facebook/esm2_t33_650M_UR50D",
    )

    parser.add_argument(
        "--lora_student",
        type=str,
        default="student_model/student_best_state.pt",
    )
    parser.add_argument("--lora_student_base", type=str, default="student_model")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = []
    if args.lora_teacher:
        results.append(evaluate_teacher(args, device))
    if args.lora_student:
        results.append(evaluate_student(args, device))

    if not results:
        raise ValueError("Provide --lora_teacher and/or --lora_student.")

    results_df = pd.DataFrame(results)
    results_df = results_df[
        ["Model", "Accuracy", "MCC", "F1", "AUC", "Precision", "Recall"]
    ]

    print("\nFINAL EVALUATION RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))
    results_df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()
