#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./data}"
mkdir -p "${DATA_ROOT}"/{mmoral,dentex,denpar,intraoral_caries,tufts}

echo "[mmoral] obtain the OralGPT release and place manifests under ${DATA_ROOT}/mmoral/manifests"
echo "[dentex] download from https://dentex.grand-challenge.org/ or https://zenodo.org/records/7812323"
echo "[denpar] download from https://zenodo.org/records/13998619"
echo "[intraoral_caries] download from https://zenodo.org/records/14827784"
echo "[tufts] request access at https://tdd.ece.tufts.edu/ then place <split>.jsonl under ${DATA_ROOT}/tufts/"

cat > "${DATA_ROOT}/MANIFEST_FORMAT.md" <<'EOF'
Each dataset under ${DATA_ROOT}/<dataset>/<split>.jsonl must be a JSONL where
every line is a single dict with these keys: image_path, label, case_id.
MMOral additionally requires a modality field within the manifest path
(see configs/data/mmoral.yaml).
EOF
