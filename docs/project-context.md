# Project Context — MATS Dental Triage

| key                  | value                                                                                              | confidence |
| -------------------- | -------------------------------------------------------------------------------------------------- | ---------- |
| project_name         | `odontocal`                                                                               | HIGH       |
| domain               | medical AI · multimodal dental imaging triage with post-hoc confidence calibration                 | HIGH       |
| framework            | PyTorch 2.1 · bfloat16 · `timm` (CNN baselines) · HuggingFace `transformers` (six 7–8B MLLMs)      | HIGH       |
| venue                | npj Digital Medicine                                                                               | HIGH       |
| primary_datasets     | 5 public sets · 29,881 images · 4 modalities · ≥6 clinical sources · 3 countries (table below)    | HIGH       |
| compute_target       | 1× NVIDIA A100 80 GB (deterministic MLLM inference + CNN training)                                | MED        |
| hparams_reference    | §4.3 L692–693, L713–716; §4.4 L730–767; §4.5 L774–797; §4.6 L799–817                              | HIGH       |
| supp_path            | none (SI PDF not provided locally; SI-only items reconstructed from prose or logged as deviation) | RESOLVED   |
| extra_signals        | Eqs 1–4 algorithmic · Eqs 5–7 metric · CLAIM-2024 adherence · deterministic MLLM decode (T=0)     | HIGH       |

## 1 — project_name
`odontocal`. Two content words from the core method (Modality-Aware Temperature Scaling) plus the application (dental triage). Derived from Title L1–3 and Methods §4.4 (MATS named in equation context).

## 2 — supp_path
None. The paper cites Supplementary Tables S1, S4, S5, S7, S7b, S8 but no companion file accompanies the submission PDF. SI-derivable values are reconstructed from prose where possible; the remainder is logged in `docs/deviations.md` with the exact line of the main-paper reference.

## 3 — domain
Medical AI · multimodal dental imaging triage with post-hoc confidence calibration. Abstract L23–26 frames it as "modality-aware calibration framework for multimodal large language model dental triage"; §1 L52–66 places it within MLLM dentistry; §4.1 L561–573 declares CLAIM-2024 adherence.

## 4 — framework
PyTorch 2.1, bfloat16 mixed precision. CNN baselines (ResNet-50, EfficientNet-B4) use `timm`; six open-source MLLMs (Qwen2.5-VL-7B, InternVL-3.5-8B, CogVLM-2-8B, LLaVA-Med-v2, HuatuoGPT-Vision-7B, MedVLM-R1) plus dental-specialised OralGPT-Omni-7B are loaded via their official HuggingFace repositories. Sources: §4.3 L686–716 ("PyTorch v2.1 with mixed precision training (bfloat16)"; "ImageNet pre-trained weights … cosine annealing over each dataset and training for 50 epochs"); Code-availability L832–834.

## 5 — venue
npj Digital Medicine. Inferred from (i) filename prefix `NPJ_DM_`; (ii) npj-style continuous line-numbering throughout; (iii) structured abstract block at L18–42; (iv) peer references [7, 12, 13, 48] are themselves npj DM articles; (v) §4.1 L572 CLAIM-2024 declaration matches npj DM submission practice.

## 6 — primary_datasets

| name             | role                                  | n_images | modality                                          | annotation                                         | license / access                                                                                                  |
| ---------------- | ------------------------------------- | -------- | ------------------------------------------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| MMOral           | primary multimodal benchmark          | 20,563   | panoramic + intraoral + periapical + cephalometric | VQA · 1.3 M instruction examples                  | "open licence" stated in paper                                              |
| DENTEX           | multi-institution panoramic           | 2,332    | panoramic                                         | FDI tooth enumeration · caries · periapical · impacted | CC BY-NC-SA · https://dentex.grand-challenge.org/ · https://zenodo.org/records/7812323                            |
| DenPAR           | periapical / periodontal              | 1,000    | periapical (IOPA)                                 | CEJ + apex + alveolar crest coordinates           | Nature Scientific Data record · https://zenodo.org/records/13998619                                               |
| Intraoral Caries | demographic shift (South Asian, 10–24)| 6,313    | intraoral photograph                              | YOLO + PASCAL VOC + COCO · κ_inter = 0.89          | Nature Scientific Data record · https://zenodo.org/records/14827784                                               |
| Tufts Dental     | panoramic + eye-tracking              | 1,000    | panoramic                                         | tooth seg · abnormality mask · ROI · gaze plots    | request-based · https://tdd.ece.tufts.edu/                                                                        |

Splits: official partitions where provided (DENTEX 705/50/250); else stratified 70/10/20 at fixed seed (§4.2 L668–670). Total test cohort > 6,200 images (§4.2 L669–671).

## 7 — compute_target
- Hardware: NVIDIA A100 80 GB (§4.3 L707).
- World size: **1** (single-node, single-GPU default). Multi-GPU reachable via explicit CLI override; DDP infrastructure is wired but not engaged by `main.yaml`.
- MLLM inference: deterministic greedy decode (`temperature = 0`); identical re-runs produce bit-identical outputs (§4.6 L815–817).
- CNN baselines: 10 random seeds (§4.6 L813–815).
- Wall-clock and storage: not reported in the paper. Upper-bound estimate for the full evaluation cohort = 6 MLLMs × ~6,200 test cases × ~8 s per case at bfloat16 on A100 ≈ 83 GPU-hours plus ~10 GPU-hours for the CNN baselines (50 epochs × 10 seeds × 4 single-task datasets at ~45 s/epoch on the smaller cohorts). Logged in deviations.

## 8 — hparams_reference

| group              | parameter                              | value                                           | source                          |
| ------------------ | -------------------------------------- | ----------------------------------------------- | ------------------------------- |
| CNN baselines      | optimiser                              | AdamW                                           | §4.3 L714                       |
| CNN baselines      | learning rate                          | 1e-4                                            | §4.3 L692                       |
| CNN baselines      | schedule                               | cosine annealing                                | §4.3 L692                       |
| CNN baselines      | epochs                                 | 50                                              | §4.3 L693                       |
| CNN baselines      | batch size                             | 32                                              | §4.3 L714                       |
| CNN baselines      | precision                              | bfloat16                                        | §4.3 L714                       |
| CNN baselines      | seeds                                  | 10                                              | §4.3 L715–716                   |
| MLLM inference     | decoding                               | greedy (T = 0)                                  | §4.6 L815–817                   |
| Calibration        | per-modality temperatures              | T_pano, T_peri, T_intra, T_ceph (independent)  | §4.4 Eq. 3 · L751–755           |
| Calibration        | fit objective                          | NLL on calibration split                        | §4.4 L733–736                   |
| Calibration        | verbalised-confidence logit transform  | ẑ = log(ĉ / (1 − ĉ))                            | §4.4 L734–736                   |
| Calibration        | ECE bins                               | B = 15                                          | §4.5 Eq. 5 · L776               |
| Calibration        | calibration-set 95% CI for T_ceph      | 1.42 – 1.68 (bootstrap, 1,000 resamples)        | §4.2 L650–653                   |
| Selective referral | threshold τ                            | 0.55                                            | §4.4 Eq. 4 · L246, L764–765     |
| Selective referral | target operating point                 | ECR ≥ 85 %, RR ≤ 20 %                           | §4.4 L764–765                   |
| Ensemble           | weights w_j(x)                         | q_j^(m)(y_max | x) / Σ_k q_k^(m)(y_max | x)     | §4.3 Eq. 1 · L694–706           |
| Statistics         | bootstrap resamples                    | B = 2000                                        | §4.6 L799–800                   |
| Statistics         | paired AUC test                        | DeLong                                          | §4.6 L800–801                   |
| Statistics         | family-wise correction                 | Holm–Bonferroni · k = 15                        | §4.6 L801–803                   |
| Statistics         | exploratory correction                 | Benjamini–Hochberg FDR · α = 0.05               | §4.6 L804–805                   |

## 9 — extra_signals
- Four numbered algorithmic equations (Eqs 1–4 in §4) and three metric equations (Eqs 5–7 in §4.5). Each is mapped to a single file in `docs/implementation-map.md`.
- Hypotheses H1–H5 are pre-registered in the Discussion (L356–378). The main eval CLI emits a confirm/refute summary; encoded in `hypotheses.yaml`.
- Five-dataset stratified subset analysis on 200 cases (Table 6): Cohen's κ = 0.73 versus expert consensus.
- Fairness gap: max AUC gap (modality) reduced 12.1 pp → 6.3 pp; max AUC gap (disease) 5.9 pp.
- Code Availability statement (L828–834): "made available from the corresponding author upon reasonable request and released publicly upon publication". The README mirrors this verbatim.
- Tufts Dental Database access is gated; the data preparation script must emit an explicit manual-step block rather than attempt automatic download.

## Resolved decisions
1. **SI handling** — proceed without the SI PDF; reconstruct from prose where possible, log the rest in deviations.
2. **World size** — single A100 80 GB default; DDP wired but inactive in `main.yaml`.

## Open items deferred to Turn 6
- MMOral SPDX licence string (paper says "open licence"; OralGPT repo has no pinned identifier).

Ready to proceed to Turn 1.
