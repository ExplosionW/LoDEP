# Structural Prototype Memory Distillation (SPMD)

SPMD is a working name for a LoDEP extension that transfers relationships to structural prototypes into a compact sequence-only student. This directory contains code and run instructions. It does not contain experiment results, checkpoints, downloaded models or structure caches.

This implementation uses a **fresh original-pretrained ESM2-8M student** and a **classification-teacher plus predicted3Di memory variant**. The teacher is the original ESM2-650M LoRA classifier. It has no 3Di auxiliary training head. Structural information enters through prototypes built from the **training split only**. This should not be described as an experiment with a teacher jointly trained on classification and3Di.

The earlier uploaded default incorrectly initialized from the already-distilled LoDEP student. It performed continued training, which does not validate fresh distillation. That default is removed here; use a new work directory. Teacher reuse remains explicit and independent of student initialization.

## Method

1. Export original LoDEP teacher logits and valid-residue pooled features for the training CSV, preserving row order.
2. Predict a 3Di state and confidence for every training residue using ProstT5 and its official CNN. No coordinate files are needed.
3. Build a 420-dimensional structural descriptor from 20 confidence-weighted state frequencies and 400 adjacent-state frequencies. Use confidence >= 0.5, square-root normalized histograms and class-conditional KMeans(seed 42). Use 32 prototypes per class where enough distinct training descriptors exist; smaller inputs reduce the number of clusters rather than creating empty prototypes.
4. Construct fixed teacher prototypes and corresponding student prototypes. Center and cosine-normalize the prototypes separately in each representation space. Match each query's distribution of similarities to the prototypes and their pairwise cosine geometry. Update touched student prototypes with EMA(alpha 0.1).
5. Initialize the backbone from the original `facebook/esm2_t6_8M_UR50D` pretrained weights. Initialize mapping and classifier randomly with seed42, and train the final ESM block plus these new heads. Do not load any previously distilled or EP-finetuned student weights. At inference, the student receives the protein sequence alone.

The objective retains the EP loss definitions, with the student initialization corrected to fresh pretrained weights:

```text
KD = KL(softmax(teacher_logits/5) || softmax(student_logits/5)) * 25
SPMD = 0.7*KD + 0.3*label_CE + 0.3*feature_MSE
       + 0.3*prototype_query_KL + 0.3*prototype_geometry_MSE
```

Two optional matched controls are included: `kd_output` uses only 0.7 KD + 0.3 CE; `kd_feature` adds 0.3 feature MSE. These retain label supervision; “output-only” describes the distillation target. Feature/prototype losses are absent from `kd_output`.

## Installation and inputs

Run from the repository root, using Python 3.10 or newer:

```bash
pip install -r spmd/requirements.txt
```

Required inputs:

- The existing LoDEP binary CSVs: `dataset/train.csv`, `dataset/validation.csv`, `dataset/test.csv`; columns `seq`(or `sequence`) and `label`(0/1). Row order is retained. There is no new split, CD-HIT, hash verification or length truncation.
- An existing original LoDEP 650M teacher adapter in `LORA_teacher/`, including its classification head. Create it with the repository's teacher-training instructions if needed. SPMD does not retrain the teacher.
- The original unadapted `facebook/esm2_t6_8M_UR50D` model, downloaded automatically or provided as a local pretrained model directory through `SPMD_STUDENT_BASE`. The task-trained model in the root `student_model/` directory must not be used to initialize this training flow.
- ProstT5(`Rostlab/ProstT5`) and the official `cnn_chkpnt/model.pt` from [the ProstT5 repository](https://github.com/mheinzinger/ProstT5). Download the CNN checkpoint and set its local path below. Hugging Face model IDs can be replaced by local model directories.

## Complete workflow on a dual-GPU node

Activate your existing training environment first. The scripts use the active environment and do not activate a machine-specific Conda path.

```bash
export SPMD_TEACHER_DIR="$PWD/LORA_teacher"
export SPMD_CNN_CHECKPOINT=/absolute/path/to/ProstT5/cnn_chkpnt/model.pt
export SPMD_WORK_DIR="$PWD/outputs/spmd_seed 42"
export SPMD_GPUS=2
bash spmd/run.sh
```

This executes data preparation, training-only full-length 3Di prediction, structural grouping, original-teacher export, then `kd_output`, `kd_feature` and `spmd`. Each training phase uses both GPUs on one node through torchrun/DDP. Set `SPMD_ARMS=spmd` to train only SPMD. For one GPU, set `SPMD_GPUS=1`; default effective batch size then changes from 32 to 16.

For one Slurm submission requesting an exclusive node and two GPUs:

```bash
export SPMD_PARTITION=kobilka-gputest  # change for your cluster
export SPMD_ARMS=spmd
bash spmd/run_slurm.sh
```

One submitted job executes the full workflow; no monitoring process is created. `SPMD_TIME_LIMIT` defaults to `2-00:00:00`. The Slurm script requests 32 CPUs and 128 GB RAM. Model downloads require network access unless the models are available locally.

Custom CSVs and optional evaluation sets can be passed to either shell entry point:

```bash
bash spmd/run.sh \
  --train-csv /path/train.csv --val-csv /path/validation.csv --test-csv /path/test.csv \
  --external-csv /path/external.csv
```

Additional optional inputs are `--pace-csv`, and the paired `--family-csv`/`--family-splits`. Family CSVs need `family_id` in the order ABC,MFS,RND,PACE,MATE,SMR(IDs 0-5). Family split JSON must map each regime(e.g.`1-shot`,`1%`,`10%`) to `train`,`val`,`test` row-index lists. The family probe fits StandardScaler+balanced LogisticRegression(C=1) on support rows only; validation/test features are not fitted. These optional inputs are not bundled or generated from the repository's extra CSVs automatically.

## Run individual stages

Keep the same `SPMD_WORK_DIR` and environment variables throughout:

```bash
python -m spmd.prepare data
python -m torch.distributed.run --standalone --nproc_per_node=2 --module spmd.predict_3di \
  --cnn-checkpoint "$SPMD_CNN_CHECKPOINT"
python -m spmd.prepare groups --structure-npz "$SPMD_WORK_DIR/structures/train.npz"
python -m torch.distributed.run --standalone --nproc_per_node=2 --module spmd.export_teacher
python -m torch.distributed.run --standalone --nproc_per_node=2 --module spmd.train_student spmd
```

Teacher export and student training skip stages whose completed output exists.3Di caches include the training sequences and lengths so row alignment is checked directly. The fresh initialization is recorded in `STUDENT_INITIALIZATION.json`. A work directory containing old student artifacts without that protocol is rejected. Choose a new work directory for a different source dataset, teacher, initialization or loss configuration. An interrupted student stage starts again from original pretrained backbone weights and new task heads on rerun; `latest_training.pt` is saved for inspection, but automatic optimizer-state resume is not implemented here.

Full-length ESM inputs and valid-residue pooling are used throughout this extension. Frozen first five block outputs are cached in GPU memory while training the final block. Longer datasets can require more VRAM than the EP setting. Lower `SPMD_BATCH_SIZE`(default 16 per GPU) or the3Di script's `--batch-size` if necessary. There is no automatic silent truncation, sequence dropping or GPU-speed profiling.

Default model paths can be changed with `SPMD_TEACHER_BASE`, `SPMD_STUDENT_BASE`, `SPMD_PROSTT5_MODEL`; set `SPMD_LOCAL_ONLY=1` for local Hugging Face files only.

## Student inference

Use this extension's inference command, which matches its valid-residue pooling:

```bash
python -m spmd.inference \
  --input-csv dataset/test.csv \
  --checkpoint "$SPMD_WORK_DIR/runs/spmd_student/student_best.pt" \
  --student-base facebook/esm2_t6_8M_UR50D \
  --output-csv outputs/spmd_predictions.csv
```

Input requires only `seq` or `sequence`; no 3Di or coordinate file is used. CUDA is selected when available, otherwise CPU. The output appends `probability_1` and `predicted_label` at threshold 0.5. The root `inference_student.py` uses the original model's pooling, so use `spmd.inference` for SPMD checkpoints.

## Configuration and output files

Seed 42; at most 20 epochs; validation-MCC early stopping(patience 3, min_delta 1e-4); checkpoint tie-breaks F1 then lower CE. Student final-block LR 1e-5, mapping/head LR 5e-5, AdamW weight decay 0.01; BF16 CUDA autocast; gradient clipping 1. Training inputs and loss settings match across the three arms. Test metrics do not select checkpoints or hyperparameters.

All generated files remain under `SPMD_WORK_DIR`:

| Path | Role |
|---|---|
|`INPUTS.json`, `data/*.csv`|Saved source paths, membership, row order and normalized sequences|
|`STUDENT_INITIALIZATION.json`|Original pretrained student source, seed42 new task heads and teacher identity|
|`structures/train.npz`|Full-length predicted training 3Di, confidence, lengths and sequences|
|`structural_groups.npz`, `STRUCTURAL_MEMORY.json`|Training-only structural group assignments and fitting protocol|
|`features/lodep_650m_teacher/train.npz`|Teacher logits/features in training-row order|
|`runs/<arm>_student/config.json`, `history.json`|Configuration and epoch validation history|
|`runs/<arm>_student/student_best.pt`|Validation-selected SPMD/KD student weights|
|`runs/<arm>_student/latest_training.pt`|Last-epoch weights, optimizer state and student memory|
|`evaluation/<model>/`|Evaluations produced when you run the code|

## Local checks

```bash
python -m unittest discover -s spmd/tests -v
bash -n spmd/run.sh spmd/run_slurm.sh
```

The CPU checks cover structural clustering, fixed teacher targets, differentiable student-memory updates, fresh pretrained initialization shared across arms, pure output-KD configuration, frozen-prefix equivalence and padding-independent pooling. They require no model download or GPU allocation. The portable package was checked on CPU; no new training run is implied by this code release. The method name remains provisional; novelty and general improvements require separate literature review and matched ablations.
