#!/usr/bin/env python3
"""Run LoDEP student model inference on CSV or FASTA input."""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


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
        return logits


def read_fasta(path):
    records = []
    current_id = None
    current_seq = []

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(current_seq)))
                current_id = line[1:].split()[0] or f"seq_{len(records)}"
                current_seq = []
            else:
                current_seq.append(line)

    if current_id is not None:
        records.append((current_id, "".join(current_seq)))

    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return pd.DataFrame(records, columns=["sequence_id", "seq"])


def read_sequences(args):
    input_path = Path(args.input_file)
    input_format = args.input_format
    if input_format == "auto":
        suffix = input_path.suffix.lower()
        input_format = "fasta" if suffix in {".fa", ".faa", ".fasta"} else "csv"

    if input_format == "fasta":
        return read_fasta(input_path)

    df = pd.read_csv(input_path)
    if args.seq_col not in df.columns:
        raise ValueError(f"CSV input must contain a '{args.seq_col}' sequence column.")

    if args.id_col and args.id_col in df.columns:
        sequence_ids = df[args.id_col].astype(str)
    else:
        sequence_ids = [f"seq_{idx}" for idx in range(len(df))]

    return pd.DataFrame(
        {
            "sequence_id": sequence_ids,
            "seq": df[args.seq_col].astype(str),
        }
    )


def load_student_model(model_dir, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    base_model = AutoModel.from_pretrained(model_dir)
    model = StudentWithMapping(
        base_model,
        student_hidden_size=checkpoint["student_hidden_size"],
        teacher_feat_dim=checkpoint["teacher_feat_dim"],
        num_labels=checkpoint["num_labels"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def build_dataloader(sequences, tokenizer, max_length, batch_size):
    encodings = tokenizer(
        sequences,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def run_inference(model, dataloader, device):
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for input_ids, attention_mask in tqdm(dataloader, desc="Predicting"):
            logits = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())

    return all_predictions, all_probabilities


def main():
    parser = argparse.ArgumentParser("Run LoDEP student model inference.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="student_predictions.csv")
    parser.add_argument("--input_format", choices=["auto", "csv", "fasta"], default="auto")
    parser.add_argument("--seq_col", type=str, default="seq")
    parser.add_argument("--id_col", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default="student_model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="student_model/student_best_state.pt",
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    inputs = read_sequences(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = load_student_model(args.model_dir, args.checkpoint, device)
    dataloader = build_dataloader(
        inputs["seq"].tolist(),
        tokenizer,
        args.max_length,
        args.batch_size,
    )
    predictions, probabilities = run_inference(model, dataloader, device)

    output_df = inputs.copy()
    output_df["predicted_label"] = predictions
    for class_idx in range(len(probabilities[0])):
        output_df[f"probability_{class_idx}"] = [
            row[class_idx] for row in probabilities
        ]

    output_df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    main()
