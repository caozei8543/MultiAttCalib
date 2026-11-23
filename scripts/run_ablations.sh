#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 ]]; then
  echo "Usage: bash scripts/run_ablations.sh DATA_ROOT SPLIT_DIR OUTPUT_DIR" >&2
  exit 2
fi
root="$1"
splits="$2"
out="$3"
for variant in full no_validity no_reciprocal_weighting no_gate constant_gate; do
  python -m multiattcalib train --config configs/paper.json --root "$root" --train "$splits/train.jsonl" --val "$splits/val.jsonl" --variant "$variant" --out "$out/$variant"
  python -m multiattcalib evaluate --root "$root" --manifest "$splits/test.jsonl" --checkpoint "$out/$variant/best.pt" --out "$out/$variant/test"
done
python -m multiattcalib plot-ablation \
  --result "No Validity=$out/no_validity/test/samples.csv" \
  --result "No Reciprocal=$out/no_reciprocal_weighting/test/samples.csv" \
  --result "No Gate=$out/no_gate/test/samples.csv" \
  --result "Full=$out/full/test/samples.csv" \
  --out "$out/fig5.png"
