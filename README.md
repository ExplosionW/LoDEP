# LoDEP

LoDEP provides a LoRA-based training and evaluation workflow for protein sequence classification with ESM-2 teacher and student models. The repository includes the LoDEP training script, evaluation script, datasets, extra few-shot family datasets, and a trained student model.

## Model Overview

![LoDEP model overview](figure/model.png)

The LoDEP framework uses an ESM-2 teacher model and a compact ESM-2 student model. The teacher is adapted with LoRA and produces both prediction logits and sequence-level feature representations. During student training, LoDEP combines response-based distillation from teacher logits with feature alignment between teacher and mapped student representations, allowing the smaller student model to learn both classification behavior and high-level protein sequence features.

## Repository Contents

```text
LoDEP-main/
|-- train_LORA.py
|-- evaluate_all.py
|-- inference_student.py
|-- figure/
|   `-- model.png
|-- dataset/
|   |-- train.csv
|   |-- validation.csv
|   |-- test.csv
|   |-- train_extra.csv
|   |-- validation_extra.csv
|   `-- test_extra.csv
|-- student_model/
|   |-- config.json
|   |-- model.safetensors
|   |-- special_tokens_map.json
|   |-- student_best_state.pt
|   |-- tokenizer_config.json
|   `-- vocab.txt
|-- requirements.txt
`-- README.md
```

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The workflow uses ESM-2 models from the `facebook/` model namespace:

- `facebook/esm2_t33_650M_UR50D` for the teacher model
- `facebook/esm2_t6_8M_UR50D` for the student model

GPU acceleration is recommended for training and evaluation.

If `--max_length` is not provided, the scripts automatically infer it from the longest tokenized sequence in the current input data, capped by the tokenizer's model limit. You can still pass `--max_length` manually to override this behavior.

## Data

The main LoDEP binary classification datasets use the following columns:

- `seq`: protein sequence
- `label`: binary class label

Files:

- `dataset/train.csv`
- `dataset/validation.csv`
- `dataset/test.csv`

The extra few-shot family datasets use:

- `seq`: protein sequence
- `family_id`: protein family label

Files:

- `dataset/train_extra.csv`
- `dataset/validation_extra.csv`
- `dataset/test_extra.csv`

The extra datasets are provided for few-shot family experiments and are not used by the main LoDEP training commands below.

## LoDEP Training Workflow

### 1. Train the LoRA Teacher

```bash
python train_LORA.py --stage teacher \
  --train_csv dataset/train.csv \
  --val_csv dataset/validation.csv \
  --tokenizer facebook/esm2_t33_650M_UR50D \
  --teacher_model facebook/esm2_t33_650M_UR50D \
  --output_dir ./LORA_teacher \
  --num_train_epochs 5 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --load_best
```

This step saves the trained teacher LoRA adapter and tokenizer files to `LORA_teacher/`.

### 2. Export Teacher Logits and Features

```bash
python train_LORA.py --stage prep_teacher_outputs \
  --train_csv dataset/train.csv \
  --tokenizer facebook/esm2_t33_650M_UR50D \
  --teacher_model facebook/esm2_t33_650M_UR50D \
  --teacher_dir ./LORA_teacher \
  --teacher_logits_out ./LORA_teacher/teacher_logits.npy \
  --teacher_feats_out ./LORA_teacher/teacher_features.npy \
  --per_device_eval_batch_size 32
```

This step generates the teacher logits and pooled feature representations used for student distillation.

### 3. Train the LoDEP Student

```bash
python train_LORA.py --stage student \
  --train_csv dataset/train.csv \
  --val_csv dataset/validation.csv \
  --tokenizer facebook/esm2_t6_8M_UR50D \
  --student_model facebook/esm2_t6_8M_UR50D \
  --teacher_logits ./LORA_teacher/teacher_logits.npy \
  --teacher_features ./LORA_teacher/teacher_features.npy \
  --output_dir_student ./LORA_student \
  --epochs 5 \
  --batch_size 32 \
  --student_lr 5e-5 \
  --alpha 0.7 \
  --beta 0.3 \
  --temperature 5.0
```

This step saves the distilled student model outputs to `LORA_student/`, including `student_best_state.pt`.

## Evaluation

Evaluate the trained student model included in this repository:

```bash
python evaluate_all.py \
  --test_csv dataset/test.csv \
  --lora_student student_model/student_best_state.pt \
  --lora_student_base student_model
```

Evaluate a newly trained LoDEP teacher and student:

```bash
python evaluate_all.py \
  --test_csv dataset/test.csv \
  --lora_teacher ./LORA_teacher \
  --lora_student ./LORA_student/student_best_state.pt \
  --lora_student_base ./LORA_student
```

The script reports Accuracy, MCC, F1, AUC, Precision, and Recall, and saves the table to `evaluation_results.csv` by default.

## Student Inference

Run inference with the trained LoDEP student model included in this repository.

CSV input must contain a sequence column named `seq` by default:

```bash
python inference_student.py \
  --input_file dataset/test.csv \
  --output_csv student_predictions.csv \
  --model_dir student_model \
  --checkpoint student_model/student_best_state.pt
```

For CSV files with a custom sequence or ID column:

```bash
python inference_student.py \
  --input_file input_sequences.csv \
  --seq_col sequence \
  --id_col protein_id \
  --output_csv student_predictions.csv
```

FASTA input is also supported:

```bash
python inference_student.py \
  --input_file input_sequences.fasta \
  --input_format fasta \
  --output_csv student_predictions.csv
```

The output CSV contains `sequence_id`, `seq`, `predicted_label`, and one probability column per class.

## Trained Student Model

The repository includes a trained LoDEP student model under `student_model/`.

- `model.safetensors`: trained student backbone weights
- `student_best_state.pt`: distilled LoDEP student state, including the mapping and classifier layers
- `config.json`: ESM model configuration
- `vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json`: tokenizer files

The teacher output caches `teacher_logits.npy` and `teacher_features.npy` are not included. They can be regenerated with the teacher export step.
