# Experiment 9: Sequence-Level Classification -- Transformer (v11)

## Objective

Instead of classifying isolated segments, model the temporal trajectory across K=16 consecutive segments from the same patient using a Transformer. The hypothesis: even though individual segments are indistinguishable across patients, the *pattern of change* over time may be informative.

## Script

- `cpc_train_v11.py` (first run)

## Architecture

- **Per-segment features**: CPC latent (64d) + CPC context (128d) + patient-normalized HRV (15d) = **207d**
- **Input projection**: Linear(207 → 256) + LayerNorm + GELU + Dropout
- **Positional encoding**: Learned `nn.Embedding(16, 256)`
- **Temporal model**: `nn.TransformerEncoder` -- 4 layers, 4 heads, 256d, 512 FFN dim, causal mask
- **Classifier**: MLP(256 → 128 → 3) on the last position's output
- **Loss**: CrossEntropyLoss with class weights

## Dataset

- `CPCSequenceDataset`: Sliding windows of K=16 consecutive SR segments from the same patient
- Sequence stride: 4 segments
- Train: 6,380 sequences (from 107 patients), Val: 1,370 sequences (from 21 patients)
- Label: time bin of the **last** segment in the window

## Training

| Parameter | Value |
|-----------|-------|
| Batch size | 128 |
| Learning rate | 3e-4 |
| Weight decay | 0.01 |
| Head-only warmup | 15 epochs |
| Fine-tune backbone LR | 3e-6 |
| Patience | 15 epochs |

## Results

| Phase | Train Acc | Train F1 | Val Acc | Val F1 |
|-------|-----------|----------|---------|--------|
| Head-only (E7, best) | 77.0% | 0.769 | 44.7% | **0.435** |
| Fine-tune (E30, final) | 96.2% | 0.962 | 40.9% | 0.401 |

**Best Val F1 Macro: 0.4353** (epoch 7, head-only phase)

**wandb**: `CPC-SeqLevel-TimeBin-v11` / run `royal-snow-1`

## Analysis

- The best result occurs **before** backbone unfreezing (epoch 7)
- After unfreezing, the model rapidly memorizes training patients (96% train) while validation degrades
- The 53pp train-val gap confirms massive overfitting
- The Transformer has too many parameters (~500K+) for only 6,380 training sequences from 107 patients

## Conclusion

The Transformer sequence model achieves a marginal improvement over v10 segment-level (43.5% vs 42.8%), suggesting the trajectory provides a tiny additional signal. However, the model's capacity is far too large for the dataset size, leading to severe overfitting. The best result comes from the frozen-backbone phase with the lightest possible training.
