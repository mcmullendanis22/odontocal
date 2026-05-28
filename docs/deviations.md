# Deviations

Empty at scaffold time. Each entry below, once populated, has the form:

```
- <short title>
  paper ref: <section / line range>
  current code: <module · symbol>
  justification: <one or two sentences>
```

## Active entries

- SI-only artefacts (Supp. Tables S1, S4, S5, S7, S7b, S8)
  paper ref: ubiquitous · §2.2 L185, §4.2 L646–648, §4.4 L766, §4.5 L796
  current code: `evaluation.*` reconstructs S5 (τ-sweep) and S7 (subgroup fairness) from main-text prose; S1 (model-list table), S4 (prompt templates), S7b (reliability bins per modality), S8 (calibrator comparison) are inferred from prose.
  justification: SI file not provided. Each reconstructed value is annotated at its call site with the main-paper line it came from.

- A100 count
  paper ref: §4.3 L707 (states "NVIDIA A100 80 GB GPUs" but not count)
  current code: `configs/experiment/main.yaml` sets `world_size: 1`
  justification: single-A100 is the most parsimonious reading of deterministic MLLM inference. DDP is wired but inactive in the shipped default.

- MMOral SPDX licence string
  paper ref: §4.2 L599–600 ("open licence")
  current code: `configs/data/mmoral.yaml` records `license: LICENSE_UNVERIFIED`
  justification: paper does not pin an SPDX identifier; downstream releaser should resolve at publication time.

- Wall-clock / storage budget
  paper ref: §4.3 L707–716 (omitted)
  current code: `README.md` records estimated 83 GPU-hours (MLLM inference) + ~10 GPU-hours (CNN baselines) derived from cohort sizes and standard A100 throughput.
  justification: paper omits these figures; the estimate is bounded and explicitly labelled as an estimate, not a measurement.
