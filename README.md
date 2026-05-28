# odontocal

**odontocal** is a research software package that calibrates the confidence of multimodal
large language models (MLLMs) for dental imaging triage. It implements modality-aware
temperature scaling (MATS), a confidence-weighted MLLM ensemble, and a selective-referral
rule, packaged with the cross-dataset evaluation harness from the npj Digital Medicine
article *How Reliable Is Your Dental AI?*. The package turns raw verbalised model
confidences into clinically usable triage probabilities across panoramic, periapical,
intraoral, and cephalometric imaging, and refers low-confidence cases to specialist review.

### Infobox

| Field | Value |
| --- | --- |
| Kind | Research code — evaluation and post-hoc calibration |
| Domain | Medical AI · multimodal dental imaging triage |
| Method | MATS · confidence-weighted ensemble · selective referral |
| Models | 7 frozen 7–8B MLLMs + ResNet-50 / EfficientNet-B4 baselines |
| Datasets | 5 public sets · 29,881 images · 4 modalities |
| Framework | PyTorch 2.1 · bfloat16 · timm · transformers |
| Configuration | Hydra / OmegaConf YAML |
| Decoding | deterministic greedy (temperature = 0) |
| Venue | npj Digital Medicine (2026) |
| Console commands | `odc-train` · `odc-infer` · `odc-fit-calibration` · `odc-evaluate` · `odc-export-onnx` |
| License | see `LICENSE` |

### Contents

1. [Background](#background)
2. [Method](#method)
3. [Package layout](#package-layout)
4. [Installation](#installation)
5. [Datasets](#datasets)
6. [Reproduction](#reproduction)
7. [Evaluation](#evaluation)
8. [Compute budget](#compute-budget)
9. [Tests](#tests)
10. [Limitations](#limitations)
11. [Ethics](#ethics)
12. [Code availability](#code-availability)
13. [See also](#see-also)
14. [References](#references)

## Background

MLLMs answer dental imaging questions fluently but report confidences that do not match
their accuracy, and the mismatch varies by imaging modality and by dataset. For a triage
setting that decision matters: an over-confident wrong answer is worse than an honest "refer
to a specialist". odontocal measures this calibration gap across five public dental datasets
and four modalities, then closes part of it with a modality-aware rescaling, an ensemble that
weights each model by its own calibrated confidence, and a referral gate that abstains when
confidence stays low.

## Method

- **Modality-aware temperature scaling (MATS).** A separate temperature `T_m` is fit per
  modality (panoramic, periapical, intraoral, cephalometric) by minimising NLL on a
  calibration split, applied to the logit-transformed verbalised confidence
  `ẑ = log(ĉ / (1 − ĉ))`. (Eq. 3)
- **Confidence-weighted ensemble.** Models are combined with weights
  `w_j(x) = q_j(y_max | x) / Σ_k q_k(y_max | x)`. (Eq. 1)
- **Selective referral.** A case is referred when calibrated confidence falls below
  `τ = 0.55`, targeting error-capture rate ≥ 85 % and referral rate ≤ 20 %. (Eq. 4)
- **Metrics.** Expected calibration error with `B = 15` bins (Eq. 5), net benefit,
  error-capture rate, referral rate, AUROC, accuracy.
- **Statistics.** Stratified bootstrap (`B = 2000`), DeLong paired-AUC test,
  Holm–Bonferroni family-wise correction (`k = 15`), Benjamini–Hochberg FDR (`α = 0.05`).

Data flow:

```
images -> MLLM / CNN predict -> MATS rescale -> confidence-weighted ensemble
       -> referral gate (τ) -> study tables + figures
```

## Package layout

The source tree is grouped by stage of the scientific method rather than by software layer.

```
src/odontocal/
  observe/     intake: dataset corpora, modality routing, prompting, splits
    corpora/   MMOral, DENTEX, DenPAR, Intraoral Caries, Tufts adapters
  model/       prediction and confidence
    backbones/ 7 MLLM wrappers + ResNet-50 / EfficientNet-B4 convnet baselines
    confidence/ temperature, ensemble, referral, Platt, isotonic, reliability, objective
  judge/       appraisal
    quality/   discrimination, calibration, clinical-utility metrics
    evidence/  bootstrap, DeLong, agreement, multiple-testing corrections
  publish/     output: study tables + figures
    tables/    primary, cross-dataset, ablation, subgroup, modality, stratified
  operate/     CNN training harness: trainer, seeding, checkpoint, distributed
  base/        shared types, protocols, registry, logging, serialization, digest
  entry/       five argparse command-line entry points
```

## Installation

pip (editable, with dev tools):

```bash
git clone <repo-url> odontocal
cd odontocal
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

conda:

```bash
conda env create -f environment.yml
conda activate odontocal
pip install -e .
```

Docker:

```bash
docker build -t odontocal:0.1 .
docker run --gpus all -it --rm -v $PWD:/workspace/odontocal odontocal:0.1
```

## Datasets

| Dataset | URL | License / access | Preprocessing | Manifest fields |
| --- | --- | --- | --- | --- |
| MMOral | OralGPT release (see paper) | open licence (paper); SPDX unverified | follow OralGPT preprocessing; emit per-modality VQA jsonl | image_path, label, case_id |
| DENTEX | https://zenodo.org/records/7812323 | CC BY-NC-SA | official 705/50/250 split; panoramic jsonl | image_path, label, case_id |
| DenPAR | https://zenodo.org/records/13998619 | Nature Scientific Data record | stratify 70/10/20 from CEJ/apex labels | image_path, label, case_id |
| Intraoral Caries | https://zenodo.org/records/14827784 | Nature Scientific Data record | binarise to {sound, caries}; stratify 70/10/20 | image_path, label, case_id |
| Tufts Dental | https://tdd.ece.tufts.edu/ | request-based (gated) | manual access; severity 1–5 -> {normal, abnormal} | image_path, label, case_id |

`bash scripts/prepare_data.sh` prints the manual access steps and writes the manifest schema
reference. Total cohort 29,881 images; test cohort > 6,200 images.

## Reproduction

CNN baselines (10 seeds):

```bash
bash scripts/launch_train.sh configs/experiment/main.yaml
# one seed directly:
odc-train --experiment configs/experiment/main.yaml --data ./data --seed 0 --ckpt-dir checkpoints/seed_0
```

MLLM inference, calibration fit, and evaluation:

```bash
for MODEL in qwen25_vl intern_vl cog_vlm2 llava_med huatuo_v medvlm_r1 oralgpt_omni; do
  odc-infer --model "$MODEL" --manifest data/eval_manifest.jsonl --output artifacts/preds_"$MODEL".jsonl
done
cat artifacts/preds_*.jsonl > artifacts/predictions.jsonl

odc-fit-calibration \
  --predictions artifacts/predictions.jsonl \
  --ground-truth artifacts/ground_truth.jsonl \
  --output artifacts/calibration.json \
  --target-capture 0.85 --referral-cap 0.20

odc-evaluate \
  --experiment configs/experiment/main.yaml \
  --predictions artifacts/predictions.jsonl \
  --ground-truth artifacts/ground_truth.jsonl \
  --out artifacts/results.json
```

Each Table-2 ablation has a dedicated experiment config, run through `odc-evaluate`:

```bash
odc-evaluate --experiment configs/experiment/ablation_no_mats.yaml ...
odc-evaluate --experiment configs/experiment/ablation_no_ensemble.yaml ...
odc-evaluate --experiment configs/experiment/ablation_no_referral.yaml ...
odc-evaluate --experiment configs/experiment/ablation_calibrator_platt.yaml ...
odc-evaluate --experiment configs/experiment/ablation_calibrator_isotonic.yaml ...
odc-evaluate --experiment configs/experiment/supplementary_tau_sweep.yaml ...
```

## Evaluation

Endpoint-to-command map with paper-reported expected values for the calibrated ensemble
(within the bootstrap CI, `B = 2000`):

| Endpoint | Command | Expected value |
| --- | --- | --- |
| Primary table — MMOral | `odc-evaluate --experiment configs/experiment/main.yaml ...` | Accuracy = 51.84 % · ECE = 0.070 |
| Primary table — DENTEX | same as above | AUROC = 0.798 · ECE = 0.085 |
| Primary table — DenPAR | same as above | AUROC = 0.762 · ECE = 0.092 |
| Primary table — Intraoral Caries | same as above | AUROC = 0.805 · ECE = 0.068 |
| Primary table — Tufts | same as above | AUROC = 0.781 · ECE = 0.088 |
| Clinical utility | same as above | net benefit = 0.188 at threshold 0.5 |
| Selective referral (τ = 0.55) | `odc-fit-calibration ... --target-capture 0.85 --referral-cap 0.20` | ECR ≈ 87.3 % · RR ≈ 18.2 % |
| Stratified subset (Table 6) | `odc-evaluate --experiment configs/experiment/main.yaml ...` | Cohen's κ = 0.73 vs expert consensus |
| Fairness gap (modality) | same as above | max AUC gap 12.1 pp -> 6.3 pp |

ONNX export of a trained CNN baseline:

```bash
odc-export-onnx --name resnet50 --n-classes 5 --ckpt checkpoints/seed_0/epoch_49.pt --out artifacts/resnet50.onnx
```

## Compute budget

| Item | Value |
| --- | --- |
| Hardware | 1× NVIDIA A100 80 GB, single node, single GPU by default |
| Parallelism | DDP wired but not engaged by `main.yaml` |
| MLLM inference | ~6,200 test images × 7 MLLMs; ~83 GPU-hours (A100-throughput estimate) |
| CNN baselines | 50 epochs × 10 seeds × 4 datasets; ~10 GPU-hours |
| Disk | ~120 GB datasets + manifests; ~5 GB per CNN seed; ~15 GB per MLLM cache |

Wall-clock and storage are not reported in the paper; the figures above are A100 estimates.

## Tests

```bash
make test     # PYTHONPATH=src pytest -q tests
make lint     # ruff check . ; black --check . ; isort --check-only .
make type     # mypy --strict src/odontocal
make smoke    # odc-evaluate --help
```

The suite spans dataset adapters (unit), MLLM wrappers (shape under dry-run), NLL gradient
(finite-difference), metrics (property-based with Hypothesis), calibration
(overfit-single-batch), training (a two-step integration smoke), evaluation (table-shape
regression), statistics (closed-form unit cases), the five command-line entry points (smoke),
and a repo-wide house-style scan that rejects banned phrases and emoji.

## Limitations

- The Supplementary Information PDF was not available locally; SI-only items are reconstructed
  from the main text where possible and otherwise recorded in `docs/deviations.md`.
- Wall-clock and storage are estimated from A100 throughput, not measured.
- The MMOral SPDX licence string is unresolved; the paper states an open licence.
- Tufts Dental access is gated; `scripts/prepare_data.sh` emits a manual-access block instead
  of downloading.

## Ethics

All five datasets are public or request-based and were collected by their original authors
under their own approvals; this package ships no patient data. The work follows the CLAIM-2024
reporting checklist (Methods §4.1). The tool is for research evaluation, not clinical
deployment; the referral gate is a study mechanism, not a medical device.

## Code availability

The article's code-availability statement reads, verbatim:

> made available from the corresponding author upon reasonable request and released publicly
> upon publication

CNN baseline weights are available on reasonable request. MLLM weights load from each model's
official HuggingFace repository (`configs/model/*.yaml`); no fine-tuning is required.
Calibration artefacts (`T_m` and `τ`) are emitted by `odc-fit-calibration`.

## See also

- DENTEX challenge — https://dentex.grand-challenge.org/
- Tufts Dental Database — https://tdd.ece.tufts.edu/

## References

[1] Chen P, Shi C, Wei B, Ling L, Yu S, Liu H, Liu B. *How Reliable Is Your Dental AI? A
Cross-Dataset Evaluation of Multimodal Large Language Models with Calibrated Confidence
Estimation for Oral Disease Triage.* npj Digital Medicine, 2026.

```bibtex
@article{odontocal_2026,
  title  = {How Reliable Is Your Dental AI? A Cross-Dataset Evaluation of Multimodal Large Language Models with Calibrated Confidence Estimation for Oral Disease Triage},
  author = {Chen, Peng and Shi, Chuyan and Wei, Bo and Ling, Lijuan and Yu, Shujuan and Liu, Huawei and Liu, Bing},
  journal= {npj Digital Medicine},
  year   = {2026}
}
```
