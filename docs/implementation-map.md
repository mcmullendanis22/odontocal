# Implementation Map

Every numbered equation, table, figure, and ablation reported in the paper is mapped below to the file/module that realises it. Module paths are relative to `odontocal` (for example `model.confidence.ensemble` is `src/odontocal/model/confidence/ensemble.py`). Notes call out the precise paper line ranges plus any decisions that affect implementation.

## Equations

| ref     | paper location              | module · symbol                                                          | notes                                                                                              |
| ------- | --------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| Eq. 1   | §4.3 L694–706               | `model.confidence.ensemble.confidence_weighted_ensemble`                 | weights normalised over the six MLLMs of the same modality; ŷ = argmax of the weighted mixture     |
| Eq. 2   | §4.4 L730–736               | `model.confidence.temperature.GlobalTS`                                  | logit transform applied before division by T whenever only a verbalised confidence is available    |
| Eq. 3   | §4.4 L737–755               | `model.confidence.temperature.ModalityAwareTS`                           | one T_m per m ∈ {pano, peri, intra, ceph}; fit by per-modality NLL on the held-out calibration set |
| Eq. 4   | §4.4 L760–767               | `model.confidence.referral.decide`                                       | accept iff q ≥ τ, refer otherwise; τ = 0.55 anchored to ECR ≥ 85 % / RR ≤ 20 %                      |
| Eq. 5   | §4.5 L774–780               | `judge.quality.calibration.expected_calibration_error`                   | B = 15 equal-width bins                                                                            |
| Eq. 6   | §4.5 L781–783               | `judge.quality.calibration.brier_score`                                  | joint discrimination + calibration penalty                                                         |
| Eq. 7   | §4.5 L785–791               | `judge.quality.clinical.net_benefit`                                     | decision-curve analysis over p_t ∈ [0.01, 0.99]                                                    |

## Tables

| ref     | paper location              | eval class · method                                                      | metrics emitted                                                                                    |
| ------- | --------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| Table 1 | §2.1 + caption L141–175     | `publish.tables.primary.PrimaryTableEval`                                | per-dataset AUC, MMOral accuracy, ECE, NB across 14 model configurations                           |
| Table 2 | §2.2 L188–219               | `publish.tables.ablation.AblationTableEval`                              | AUC, ECE, NB, ECR, RR for each ablation row                                                        |
| Table 3 | §2.2 L220–242 + caption     | `publish.tables.modality.ModalityTableEval`                             | per-modality uncalibrated / global-TS / MATS ECE plus Δ                                            |
| Table 4 | §2.3 caption + L221–235     | `publish.tables.cross_dataset.CrossDatasetTableEval`                    | per-target-dataset AUC, ΔAUC, ECE                                                                   |
| Table 5 | §2.4 L243–262               | `publish.tables.subgroup.SubgroupTableEval`                             | AUC + ECE + ECR + RR per disease category, modality, and dataset; fairness gap summary             |
| Table 6 | §2.5 L263–296               | `publish.tables.stratified.StratifiedSubsetEval`                        | 200-case concordance, Cohen's κ, Δ vs majority-class clinical baseline                             |

## Ablations (configs/experiment/)

| file                                   | rows removed (Table 2)                              | hypotheses touched |
| -------------------------------------- | --------------------------------------------------- | ------------------ |
| `main.yaml`                            | full system                                         | H1, H2, H3         |
| `ablation_no_mats.yaml`                | strip per-modality T; use global T (Eq. 2)          | H2                 |
| `ablation_no_ensemble.yaml`            | best calibrated single MLLM (no Eq. 1 weighting)    | H1                 |
| `ablation_no_referral.yaml`            | omit Eq. 4 routing; accept all                      | H1, H3             |
| `ablation_calibrator_platt.yaml`       | Platt sigmoid in place of TS                        | side analysis      |
| `ablation_calibrator_isotonic.yaml`    | isotonic regression in place of TS                  | side analysis      |
| `supplementary_tau_sweep.yaml`         | τ ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}             | reconstructs Supp. Table S5                                                                |

## Figures

| ref     | paper location                                            | producer                                                                                                                                                            |
| ------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fig. 1  | Visual heterogeneity panel L132 area                      | `publish.figures.fig1_modality_panel` (visual summary; the paper figure is drawn by hand — code emits the numbers behind the calibration-gap row)                   |
| Fig. 2  | Method overview L152 area                                 | descriptive only                                                                                                                                                    |
| Fig. 3  | Pipeline diagram L600 area                                | descriptive only                                                                                                                                                    |
| Fig. 4  | MATS architecture L740 area                               | `publish.figures.fig4_routing_diagram` — emits the modality-routing → T_m → softmax flow as a Graphviz DOT artefact                                                 |
| Fig. 5  | Mathematical foundations L756 area                        | `publish.figures.fig5_math_principles` — reproduces (A) temperature flattening contour, (D) reliability diagram, (E) decision curve from raw outputs                |

## Statistics

| ref               | paper location              | module                                                                                                                                                            |
| ----------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Bootstrap CI      | §4.6 L799–800               | `judge.evidence.bootstrap.stratified_bootstrap_ci` — B = 2000 stratified resamples                                                                                 |
| DeLong            | §4.6 L800–801               | `judge.evidence.delong.delong_paired_auc_test` — closed-form covariance using Sun & Xu (2014) formulation                                                          |
| Holm–Bonferroni   | §4.6 L801–803               | `judge.evidence.corrections.holm_bonferroni` — k = 15 primary comparisons                                                                                          |
| BH-FDR            | §4.6 L804–805               | `judge.evidence.corrections.benjamini_hochberg` — α = 0.05 on exploratory analyses                                                                                 |
| Cohen's κ         | §2.5 + Table 6              | `judge.evidence.agreement.cohens_kappa`                                                                                                                            |

## Hypotheses → eval mapping

| id  | text (paraphrase)                                                                  | confirmed by                                                                       |
| --- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| H1  | calibrated NB ≥ uncalibrated NB (improvement ≥ pre-specified threshold)            | Table 2 NB column · main vs No-calibration row                                     |
| H2  | modality/temperature dependent calibration outperforms global TS in ECE            | Table 2 ECE column · main vs Modality-aware TS-only row                            |
| H3  | ECR ≥ 85 % at RR ≤ 20 %                                                            | Table 2 last column · main row                                                     |
| H4  | equipment-driven AUC variance > demographic-driven AUC variance                    | Table 4 ΔAUC column (qualitative, hypothesis-generating per §2.3 L236–242)         |
| H5  | medical-specialised MLLMs underperform general MLLMs on dental tasks               | Table 1 medical-specialised rows vs open-source-general rows                       |

## Data adapters

| dataset            | adapter module                                       | label scheme                                       |
| ------------------ | ---------------------------------------------------- | --------------------------------------------------- |
| MMOral             | `observe.corpora.mmoral.MMOralVQA`                  | four diagnostic categories from VQA pairs            |
| DENTEX             | `observe.corpora.dentex.DentexPanoramic`            | caries · deep caries · periapical · impacted        |
| DenPAR             | `observe.corpora.denpar.DenPARPeriapical`           | CEJ/apex ratio → periodontal staging                |
| Intraoral Caries   | `observe.corpora.intraoral.IntraoralCaries`         | caries vs sound                                     |
| Tufts              | `observe.corpora.tufts.TuftsPanoramic`              | abnormality severity 1–5 → coarse 2-class           |

## MLLM wrappers

| family             | wrapper                                                | confidence path                                                                          |
| ------------------ | ------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| Qwen2.5-VL-7B      | `model.backbones.qwen25_vl.Qwen25VLWrapper`            | verbalised confidence parsed from structured JSON reply                                  |
| InternVL-3.5-8B    | `model.backbones.intern_vl.InternVLWrapper`            | verbalised confidence                                                                    |
| CogVLM-2-8B        | `model.backbones.cog_vlm2.CogVLM2Wrapper`              | verbalised confidence                                                                    |
| LLaVA-Med-v2       | `model.backbones.llava_med.LlavaMedWrapper`            | verbalised confidence (medical-specialised)                                              |
| HuatuoGPT-Vision-7B| `model.backbones.huatuo_v.HuatuoVisionWrapper`         | verbalised confidence (Chinese-medical-specialised)                                      |
| MedVLM-R1          | `model.backbones.medvlm_r1.MedVLMR1Wrapper`            | verbalised confidence                                                                    |
| OralGPT-Omni-7B    | `model.backbones.oralgpt_omni.OralGPTOmniWrapper`      | token-level + verbalised; uses logits where exposed by official inference code           |
