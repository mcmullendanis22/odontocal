# Repo Plan

## Directory tree (final)

```
odontocal/
├── README.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── Dockerfile
├── .gitignore
├── configs/
│   ├── hypotheses.yaml
│   ├── model/
│   │   ├── cogvlm2.yaml
│   │   ├── efficientnet_b4.yaml
│   │   ├── huatuogpt_v.yaml
│   │   ├── internvl35.yaml
│   │   ├── llava_med.yaml
│   │   ├── medvlm_r1.yaml
│   │   ├── oralgpt_omni.yaml
│   │   ├── qwen25vl.yaml
│   │   └── resnet50.yaml
│   ├── data/
│   │   ├── denpar.yaml
│   │   ├── dentex.yaml
│   │   ├── intraoral_caries.yaml
│   │   ├── mmoral.yaml
│   │   └── tufts.yaml
│   ├── train/
│   │   ├── calibration_fit.yaml
│   │   └── cnn_baseline.yaml
│   └── experiment/
│       ├── _smoke.yaml
│       ├── ablation_calibrator_isotonic.yaml
│       ├── ablation_calibrator_platt.yaml
│       ├── ablation_no_ensemble.yaml
│       ├── ablation_no_mats.yaml
│       ├── ablation_no_referral.yaml
│       ├── main.yaml
│       └── supplementary_tau_sweep.yaml
├── src/odontocal/
│   ├── __init__.py
│   ├── py.typed
│   ├── observe/
│   │   ├── __init__.py
│   │   ├── corpora/
│   │   │   ├── __init__.py
│   │   │   ├── denpar.py
│   │   │   ├── dentex.py
│   │   │   ├── intraoral.py
│   │   │   ├── mmoral.py
│   │   │   └── tufts.py
│   │   ├── prompting.py
│   │   ├── routing.py
│   │   └── splits.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── backbones/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── cog_vlm2.py
│   │   │   ├── convnet.py
│   │   │   ├── huatuo_v.py
│   │   │   ├── intern_vl.py
│   │   │   ├── llava_med.py
│   │   │   ├── medvlm_r1.py
│   │   │   ├── oralgpt_omni.py
│   │   │   └── qwen25_vl.py
│   │   └── confidence/
│   │       ├── __init__.py
│   │       ├── ensemble.py
│   │       ├── isotonic.py
│   │       ├── objective.py
│   │       ├── platt.py
│   │       ├── referral.py
│   │       ├── reliability.py
│   │       └── temperature.py
│   ├── judge/
│   │   ├── __init__.py
│   │   ├── quality/
│   │   │   ├── __init__.py
│   │   │   ├── calibration.py
│   │   │   ├── clinical.py
│   │   │   └── discrimination.py
│   │   └── evidence/
│   │       ├── __init__.py
│   │       ├── agreement.py
│   │       ├── bootstrap.py
│   │       ├── corrections.py
│   │       └── delong.py
│   ├── publish/
│   │   ├── __init__.py
│   │   ├── figures.py
│   │   └── tables/
│   │       ├── __init__.py
│   │       ├── ablation.py
│   │       ├── cross_dataset.py
│   │       ├── modality.py
│   │       ├── primary.py
│   │       ├── stratified.py
│   │       └── subgroup.py
│   ├── operate/
│   │   ├── __init__.py
│   │   ├── checkpoint.py
│   │   ├── distributed.py
│   │   ├── seeding.py
│   │   └── trainer.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── digest.py
│   │   ├── logging.py
│   │   ├── protocols.py
│   │   ├── registry.py
│   │   ├── serialization.py
│   │   └── types.py
│   └── entry/
│       ├── __init__.py
│       ├── evaluate.py
│       ├── export_onnx.py
│       ├── fit_calibration.py
│       ├── infer.py
│       └── train.py
├── scripts/
│   ├── launch_eval.sh
│   ├── launch_train.sh
│   └── prepare_data.sh
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   ├── test_backbones_shape.py
│   ├── test_confidence_overfit_single_batch.py
│   ├── test_corpora_unit.py
│   ├── test_entry_smoke.py
│   ├── test_evidence_unit.py
│   ├── test_house_style.py
│   ├── test_objective_gradient.py
│   ├── test_operate_smoke.py
│   ├── test_quality_properties.py
│   └── test_tables_regression.py
├── docs/
│   ├── deviations.md
│   ├── implementation-map.md
│   ├── project-context.md
│   └── repo-plan.md
└── assets/
    └── (figures, schema diagrams)
```

The package is grouped by stage of the scientific method: `observe` (data intake), `model`
(prediction + confidence), `judge` (metrics + statistical evidence), `publish` (tables +
figures), `operate` (CNN training harness), `base` (shared infrastructure), `entry` (command
line). Registry keys (`qwen25_vl`, `mmoral`, `modality_aware_ts`, ...) and config group names
are stable and independent of file names.

## Dependency list (pinned)

Runtime:
- `torch == 2.1.2`
- `torchvision == 0.16.2`
- `timm == 0.9.16`
- `transformers == 4.44.2`
- `accelerate == 0.33.0`
- `numpy == 1.26.4`
- `scipy == 1.13.1`
- `scikit-learn == 1.4.2`
- `pandas == 2.2.2`
- `pyarrow == 16.1.0`
- `omegaconf == 2.3.0`
- `hydra-core == 1.3.2`
- `pyyaml == 6.0.2`
- `pillow == 10.4.0`
- `matplotlib == 3.9.2`
- `tqdm == 4.66.4`
- `requests == 2.32.3`

Development:
- `pytest == 8.3.3`
- `pytest-cov == 5.0.0`
- `hypothesis == 6.112.1`
- `ruff == 0.6.9`
- `black == 24.8.0`
- `isort == 5.13.2`
- `mypy == 1.11.2`
- `pre-commit == 3.8.0`

## Expected test coverage

- 10 dedicated test files (one per stage, plus an end-to-end smoke and a house-style scan).
- Unit/property/shape tests are deterministic and run in CI; the integration test
  (`test_operate_smoke.py`) runs 2 training steps on `configs/experiment/_smoke.yaml` and
  asserts loss does not increase.
- Coverage target: ≥ 80 % on `src/odontocal/model/confidence`, `judge/quality`,
  `judge/evidence` (the load-bearing science); ≥ 50 % on `publish/tables`.
