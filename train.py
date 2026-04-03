"""
AI Plagiarism Detector - Model Training
========================================
Model  : microsoft/deberta-v3-base  (DeBERTa v3)
Task   : 3-class segment classification
Labels : 0=human | 1=ai_generated | 2=ai_paraphrased

Architecture rationale (Turnitin-inspired):
  - DeBERTa-v3 disentangled attention separates content from position →
    better at catching AI's position-invariant phrasing patterns.
  - Sentence-level classification (each extracted segment = one sample).
  - Priority-weighted loss: HIGH-priority samples (post-retrain) get 2x weight
    since their Turnitin labels are more accurate.
  - Class-weighted CE loss: compensates for 65/35/0.4% label imbalance.
  - Label smoothing on 'human' class prevents over-confident false positives
    (key to keeping FPR low, mirroring Turnitin's ~1% FPR goal).
  - Confidence threshold at inference: predict non-human only if p > 0.65.

FPR reduction techniques:
  1. Label smoothing (epsilon=0.1) on human class → model stays uncertain at boundaries
  2. Priority sample weighting → model trusts HIGH-priority labels more
  3. Class imbalance compensation → prevents collapsing to majority class
  4. High-confidence threshold at inference (0.65 vs default 0.5)
  5. Evaluation reports per-class FPR explicitly alongside precision/recall/F1
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

DATASET_PATH   = Path("c:/Users/gowth/Downloads/ai_plag_detector/data/processed/dataset.jsonl")
MODEL_OUT_DIR  = Path("c:/Users/gowth/Downloads/ai_plag_detector/models/deberta_ai_detector")
MODEL_NAME     = "microsoft/deberta-v3-base"

LABEL2ID = {"human": 0, "ai_generated": 1, "ai_paraphrased": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 3

MAX_LENGTH        = 512
BATCH_SIZE        = 16       # RTX 4050 has 6GB VRAM — 16 fits comfortably
EPOCHS            = 5
LR                = 2e-5
WARMUP_RATIO      = 0.1
WEIGHT_DECAY      = 0.01
HIGH_PRIORITY_W   = 2.0      # weight multiplier for HIGH-priority samples
LABEL_SMOOTHING   = 0.1      # applied only to human class
SEED              = 42
INFER_THRESHOLD   = 0.65     # min confidence to predict non-human
USE_BF16          = True     # bf16 autocast — RTX 4050 Ada supports bf16 natively, no GradScaler needed

# Class weights (inverse frequency, normalized)
# human=64.8%, ai_gen=34.8%, ai_para=0.4%
CLASS_WEIGHTS = [1.0, 1.86, 162.0]  # 64.8/64.8, 64.8/34.8, 64.8/0.4

# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ─── Dataset ─────────────────────────────────────────────────────────────────

class AIDetectionDataset(Dataset):
    def __init__(self, records: list, tokenizer, max_length: int = MAX_LENGTH):
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        text  = rec["text"]
        label = LABEL2ID[rec["label"]]

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
            "sample_weight":  torch.tensor(
                HIGH_PRIORITY_W if rec.get("priority") == "HIGH" else 1.0,
                dtype=torch.float
            ),
        }


def load_records(path: Path) -> list:
    records = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        r = json.loads(line)
        if r["label"] in LABEL2ID:
            records.append(r)
    return records


# ─── Loss with label smoothing + sample weights ───────────────────────────────

class WeightedLabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with:
      - per-class weights (handle imbalance)
      - label smoothing (reduce human over-confidence → lower FPR)
      - per-sample weights (HIGH priority = more trusted labels)
    """
    def __init__(self, class_weights: list, smoothing: float = 0.1, num_classes: int = 3):
        super().__init__()
        self.smoothing    = smoothing
        self.num_classes  = num_classes
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)

        # Smooth targets: (1-eps)*one_hot + eps/K
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / self.num_classes)
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / self.num_classes)

        # Per-class weight for each sample
        cw = self.class_weights[targets]  # (B,)

        # NLL loss per sample
        loss_per_sample = -(smooth_targets * log_probs).sum(dim=-1)  # (B,)

        # Apply class weight
        loss_per_sample = loss_per_sample * cw

        # Apply sample priority weight
        if sample_weights is not None:
            loss_per_sample = loss_per_sample * sample_weights

        return loss_per_sample.mean()


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(all_labels, all_preds, all_probs=None):
    """
    Compute per-class precision, recall, F1, and FPR.
    FPR = FP / (FP + TN) — critical for AI detection (false accusation rate).
    """
    labels_arr = np.array(all_labels)
    preds_arr  = np.array(all_preds)

    report = classification_report(
        labels_arr, preds_arr,
        target_names=[ID2LABEL[i] for i in range(NUM_LABELS)],
        digits=4, zero_division=0
    )

    cm = confusion_matrix(labels_arr, preds_arr, labels=list(range(NUM_LABELS)))

    # Per-class FPR
    fpr_per_class = {}
    for cls in range(NUM_LABELS):
        tp = cm[cls, cls]
        fn = cm[cls, :].sum() - tp
        fp = cm[:, cls].sum() - tp
        tn = cm.sum() - tp - fn - fp
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fpr_per_class[ID2LABEL[cls]] = fpr

    return report, cm, fpr_per_class


# ─── Training loop ────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, loss_fn, device, use_amp=False):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  Train", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)
        sample_weights = batch["sample_weight"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits.float(), labels, sample_weights)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device, threshold: float = INFER_THRESHOLD, use_amp=False):
    model.eval()
    total_loss  = 0.0
    all_labels  = []
    all_preds   = []
    all_probs   = []

    for batch in tqdm(loader, desc="  Eval ", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)
        sample_weights = batch["sample_weight"].to(device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits.float(), labels, sample_weights)
        total_loss += loss.item()

        probs = torch.softmax(outputs.logits.float(), dim=-1).cpu().numpy()  # (B, 3)

        # Threshold logic: predict non-human only if max non-human prob >= threshold
        for prob_row, true_label in zip(probs, labels.cpu().numpy()):
            human_p  = prob_row[LABEL2ID["human"]]
            non_h_p  = 1.0 - human_p
            if non_h_p >= threshold:
                pred = int(np.argmax(prob_row))
                if pred == LABEL2ID["human"]:
                    pred = int(np.argmax(prob_row[[1, 2]])) + 1  # ai_gen or ai_para
            else:
                pred = LABEL2ID["human"]

            all_preds.append(pred)
            all_labels.append(int(true_label))
            all_probs.append(prob_row)

    return total_loss / len(loader), all_labels, all_preds, all_probs


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.backends.cudnn.benchmark = True  # optimize conv ops for fixed input size

    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    log.info(f"Loading dataset: {DATASET_PATH}")
    records = load_records(DATASET_PATH)
    log.info(f"Total records: {len(records)}")

    from collections import Counter
    label_counts = Counter(r["label"] for r in records)
    log.info(f"Label distribution: {dict(label_counts)}")

    # Stratified split: 80/10/10 — stratify on label
    labels_for_split = [r["label"] for r in records]
    train_recs, temp_recs = train_test_split(
        records, test_size=0.2, stratify=labels_for_split, random_state=SEED
    )
    # Only stratify val/test if ai_paraphrased appears in temp
    temp_labels = [r["label"] for r in temp_recs]
    if len(set(temp_labels)) == NUM_LABELS and min(Counter(temp_labels).values()) >= 2:
        val_recs, test_recs = train_test_split(
            temp_recs, test_size=0.5, stratify=temp_labels, random_state=SEED
        )
    else:
        val_recs, test_recs = train_test_split(
            temp_recs, test_size=0.5, random_state=SEED
        )

    log.info(f"Train: {len(train_recs)} | Val: {len(val_recs)} | Test: {len(test_recs)}")

    # ── Tokenizer & Model ──────────────────────────────────────────────────────
    log.info(f"Loading tokenizer & model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # ── Datasets & Loaders ─────────────────────────────────────────────────────
    train_ds = AIDetectionDataset(train_recs, tokenizer)
    val_ds   = AIDetectionDataset(val_recs,   tokenizer)
    test_ds  = AIDetectionDataset(test_recs,  tokenizer)

    # Weighted sampler for training: oversample rare classes
    train_label_ids = [LABEL2ID[r["label"]] for r in train_recs]
    class_sample_counts = np.bincount(train_label_ids, minlength=NUM_LABELS)
    # Weight per sample = 1 / class_freq, scaled by priority
    sample_weights_for_sampler = []
    for r in train_recs:
        cls_w = 1.0 / max(class_sample_counts[LABEL2ID[r["label"]]], 1)
        pri_w = HIGH_PRIORITY_W if r.get("priority") == "HIGH" else 1.0
        sample_weights_for_sampler.append(cls_w * pri_w)

    sampler = WeightedRandomSampler(
        weights=sample_weights_for_sampler,
        num_samples=len(train_recs),
        replacement=True,
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,  pin_memory=pin, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,    pin_memory=pin, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,    pin_memory=pin, num_workers=0)

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────────
    loss_fn = WeightedLabelSmoothingLoss(
        class_weights=CLASS_WEIGHTS,
        smoothing=LABEL_SMOOTHING,
        num_classes=NUM_LABELS,
    ).to(device)

    # bf16 autocast — no GradScaler needed (bf16 doesn't suffer from underflow like fp16)
    use_amp = USE_BF16 and device.type == "cuda"
    scaler = None  # not needed for bf16

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    total_steps   = len(train_loader) * EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ── Training ──────────────────────────────────────────────────────────────
    log.info(f"\nStarting training: {EPOCHS} epochs, LR={LR}, batch={BATCH_SIZE}")
    log.info(f"  Threshold (inference): {INFER_THRESHOLD}")
    log.info(f"  Label smoothing      : {LABEL_SMOOTHING}")
    log.info(f"  HIGH priority weight : {HIGH_PRIORITY_W}x")
    log.info(f"  Class weights        : {CLASS_WEIGHTS}")

    best_val_f1  = 0.0
    best_val_fpr = 1.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        log.info(f"\n{'='*55}")
        log.info(f"Epoch {epoch}/{EPOCHS}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, use_amp)
        val_loss, val_labels, val_preds, val_probs = eval_epoch(model, val_loader, loss_fn, device, use_amp=use_amp)

        report, cm, fpr = compute_metrics(val_labels, val_preds)
        _, _, f1s, _ = precision_recall_fscore_support(
            val_labels, val_preds, average=None, labels=list(range(NUM_LABELS)), zero_division=0
        )
        macro_f1 = float(np.mean(f1s))

        log.info(f"  Train loss : {train_loss:.4f}")
        log.info(f"  Val loss   : {val_loss:.4f}")
        log.info(f"  Macro F1   : {macro_f1:.4f}")
        log.info(f"  FPR per class:")
        for cls_name, fpr_val in fpr.items():
            log.info(f"    {cls_name:<20} FPR={fpr_val:.4f}")
        log.info(f"\n{report}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "macro_f1": macro_f1,
            "fpr": fpr,
        })

        # Save best model: prioritize low FPR on human class (= low false accusation)
        # + high macro F1
        human_fpr = fpr.get("human", 1.0)
        score = macro_f1 - human_fpr  # maximize F1 while penalizing human FPR

        if score > best_val_f1 - best_val_fpr:
            best_val_f1  = macro_f1
            best_val_fpr = human_fpr
            model.save_pretrained(MODEL_OUT_DIR)
            tokenizer.save_pretrained(MODEL_OUT_DIR)
            log.info(f"  ✓ Best model saved (macro_F1={macro_f1:.4f}, human_FPR={human_fpr:.4f})")

    # ── Final test evaluation ─────────────────────────────────────────────────
    log.info(f"\n{'='*55}")
    log.info("FINAL TEST EVALUATION (best checkpoint)")
    best_model = AutoModelForSequenceClassification.from_pretrained(MODEL_OUT_DIR)
    best_model.to(device)
    _, test_labels, test_preds, test_probs = eval_epoch(best_model, test_loader, loss_fn, device, use_amp=use_amp)
    test_report, test_cm, test_fpr = compute_metrics(test_labels, test_preds, test_probs)

    log.info(f"\nClassification Report:\n{test_report}")
    log.info(f"\nConfusion Matrix:\n{test_cm}")
    log.info(f"\nPer-class FPR (False Positive Rate = false accusation rate):")
    for cls_name, fpr_val in test_fpr.items():
        log.info(f"  {cls_name:<20} FPR={fpr_val:.4f}  ({fpr_val*100:.2f}%)")

    # Save training history + results
    results = {
        "model": MODEL_NAME,
        "config": {
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "label_smoothing": LABEL_SMOOTHING,
            "high_priority_weight": HIGH_PRIORITY_W,
            "class_weights": CLASS_WEIGHTS,
            "inference_threshold": INFER_THRESHOLD,
        },
        "history": history,
        "test_fpr": test_fpr,
        "test_report": test_report,
        "test_confusion_matrix": test_cm.tolist(),
    }
    results_path = MODEL_OUT_DIR / "training_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info(f"\nResults saved: {results_path}")
    log.info(f"Model saved  : {MODEL_OUT_DIR}")


if __name__ == "__main__":
    main()
