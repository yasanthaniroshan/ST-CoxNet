# Experiment 2: AFib vs SR Binary Classification

## Objective

Use the pretrained CPC encoder to classify individual segments as Atrial Fibrillation (AFib) or Sinus Rhythm (SR). This validates whether the CPC representations are useful for a downstream clinical task.

## Script

- `cpc_train_v4.py` (initial), `cpc_train_v8.py` (final)

## Architecture

- **Backbone**: Pretrained CPC encoder (frozen initially, then fine-tuned)
- **Classification Head**: `AFibClassificationHead` -- MLP on CPC context + latent features
- **Loss**: BCEWithLogitsLoss

## Dataset

- `CPCClassificationDataset`: Filters for pure SR (label=-1) and pure AFib (label=1) segments
- Mixed segments (label=0) excluded
- Same patient-level train/val split as CPC pretraining

## Training Protocol

1. **Phase 1 (head-only)**: Freeze encoder, train classification head only
2. **Phase 2 (fine-tune)**: Unfreeze encoder with reduced learning rate

## Results

| Metric | Train | Validation |
|--------|-------|------------|
| Accuracy | ~95% | ~93% |
| F1 Score (macro) | ~95% | ~92.7% |
| Precision | ~95% | ~93% |
| Recall | ~95% | ~92% |

## Conclusion

The CPC encoder learns representations that are highly discriminative for AFib vs SR classification, achieving **92.7% F1** on held-out patients. This confirms the encoder captures meaningful cardiac rhythm features. However, this task (distinguishing AFib from SR) is relatively easy -- the RR patterns are fundamentally different between the two rhythms.
