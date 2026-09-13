# Online Urdu Text-Line Recognition — Reproduction

Independent reproduction of results from *"Online Urdu Text-Line Recognition by Bridging Stroke Dynamics and Offline Representations"*, using the authors' released code and the public OUHD-L dataset.

Repo: [dll-ncai/Online-Urdu-HWR](https://github.com/dll-ncai/Online-Urdu-HWR)

## Results

| Configuration | Metric | Paper (5-seed mean, full convergence) | This reproduction (single seed) |
|---|---|---|---|
| Zero-shot (offline backbone, no fine-tuning) | Test CER | 10.65% | **10.69%** |
| Ink-only fine-tune | Test CER | 4.21% (±0.09%) | **4.30%** |
| Full 11-channel fusion | Test CER | 3.66% (±0.04%) | **4.44%** (6 epochs, incomplete — see note below) |

Zero-shot and ink-only fine-tuning are both close, single-seed matches to the paper's reported figures. The fusion result is reported honestly as incomplete: the run was terminated by a hard 12-hour session limit on the free compute tier used for this reproduction (see Setup), after only 6 of the paper's intended training epochs. Validation CER was still actively improving at the point of termination, with no sign of plateau, so the gap to the paper's converged number is attributed to training duration, not a methodological difference.

One additional discrepancy worth noting: the released ink-only fine-tuning script uses `ctc_weight=0.5, ce_weight=0.5`, while the paper's stated general training recipe is `ctc_weight=0.8, ce_weight=0.2` (which the released fusion script does use). This reproduction ran the ink-only stage exactly as the released script specifies, not as separately re-tuned to match the paper's stated recipe, and reports the result under that condition.

## Bugs found in the released codebase

### 1. Tokenizer special-token registration bug (critical)

`tokeniser.py`'s `get_tokenizer()` sets the tokenizer's special tokens via direct attribute assignment:
```python
tokenizer.bos_token = '<s>'
tokenizer.eos_token = '</s>'
tokenizer.pad_token = '<pad>'
tokenizer.unk_token = '<unk>'
```
This sets the token *values* correctly, but does not register them with the tokenizer's internal special-tokens tracking. As a result, `tokenizer.batch_decode(..., skip_special_tokens=True)` silently fails to strip padding/special tokens from decoded output during evaluation.

**Effect:** reference strings retained hundreds of literal `<pad>` tokens after decoding, while predicted strings did not (predictions are naturally shorter than the padded max sequence length). Comparing these directly produced a character error rate of ~90.6% on a zero-shot evaluation, despite the underlying predictions being visually correct on manual inspection.

**Fix:**
```python
tokenizer.add_special_tokens({
    'bos_token': '<s>', 'eos_token': '</s>',
    'pad_token': '<pad>', 'unk_token': '<unk>'
})
```
Safe to apply without retraining — the tokens already exist in the vocabulary at their original fixed ids; this only corrects how they're registered, not their values.

*Note: this may already be patched in the current version of the upstream repo — worth checking the latest commit before assuming it's still present.*

### 2. Undocumented required keyword argument

`evaluate()` in `train_uhwr_fine_tune_camera_ready.py` requires a `data="img_stroke"` keyword argument to know which key of the `pixel_values` dict to extract. This isn't documented at the call site and omitting it raises `AttributeError: 'dict' object has no attribute 'to'` rather than a clear error about the missing argument.

### 3. Hardcoded local paths

`ROOT`, `DATA_ROOT`, and `decoder_path` in the training scripts are hardcoded Windows absolute paths (e.g. `C:\AliCode\...`), preventing the scripts from running as-is on any other machine or OS without manual editing. Not a logic bug, but a real portability barrier for anyone trying to reproduce results without first locating and patching these.

## Setup

- Compute: Kaggle free tier (single/dual NVIDIA T4), no other compute used for this paper.
- Dataset: OUHD-L, public Zenodo release (Core + Generative archives).
- Single seed throughout (seed 42, per the released scripts' default).
- Zero-shot and ink-only runs completed to their natural stopping point (early stopping). Fusion run was time-limited by Kaggle's 12-hour session cap, not by convergence or early stopping.

## How to run

1. Clone the [original repo](https://github.com/dll-ncai/Online-Urdu-HWR) and apply the tokenizer fix above to `tokeniser.py`.
2. Download OUHD-L (Core archive is sufficient for zero-shot/ink-only; both Core and Generative are needed for fusion) and merge into a single root directory (the released manifests reference paths spanning both archives as if extracted into one tree).
3. Rebuild `train_leakproof.csv` / `val_leakproof.csv` / `test_leakproof.csv` if not already present — no column renaming is needed if using the public manifest as-is, its columns already match what the dataset loader expects.
4. Zero-shot: load `best_model_uhwr_icdar.pt` directly into `JointModel` and evaluate on the test split, no training.
5. Ink-only: run `train_uhwr_fine_tune_camera_ready.py`, passing `data="img_stroke"` explicitly wherever `evaluate()` is called.
6. Fusion: run `train_uhwr_online_camera_ready.py` with the 11 auxiliary channels (`img_dx, img_dy, img_sin_theta, img_cos_theta, img_curvature, img_speed, img_acceleration, img_time_norm, img_pressure, img_x_tilt, img_y_tilt`).
