#!/usr/bin/env bash
# Evaluate one or more trained models on the test split of each original dataset,
# then print a single comparison table (dataset-level IoU / Precision / Recall / F1).
#
# Run from the project root:
#     bash scripts/eval_per_dataset.sh
#
# Edit the MODELS list below: "label=path/to/best_model.pth".
# Add your single-dataset baselines next to the merged model to get the full comparison.

set -euo pipefail

DATASET_ROOT=/mnt/nas151/sar/Footprint/datasets
TILE_DIR=tiles_256
SPLIT=val            # Inria's official test set has no GT: if Inria prints no metrics, use SPLIT=val for it
ARCH=unetLL
OUT_ROOT=experiments/per_dataset_eval

# Original datasets only (NOT the _s128 copy: test tiles must not overlap)
DATASETS=(WHUBuildingDataset InriaBuildingDataset MassachusettsBuildingDataset)

MODELS=(
  "merged=runs/Merged_WHU_Inria_Mass/unetLL_bce_dim256_n68500_bs32/best_model.pth"
  "whu_only=runs/WHUBuildingDataset/unetLL_bce_dim256_n47088_bs32/best_model.pth"
  "inria_only=runs/InriaBuildingDataset/unetLL_bce_dim256_n50540_bs32/best_model.pth"
  "mass_only=runs/MassachusettsBuildingDataset_s128/unetLL_bce_dim256_n13700_bs32/best_model.pth"
)

for entry in "${MODELS[@]}"; do
  label="${entry%%=*}"
  model="${entry#*=}"
  for ds in "${DATASETS[@]}"; do
    out="$OUT_ROOT/$label/$ds"
    if [[ -f "$out/metrics_summary.json" ]]; then
      echo "[skip] $label on $ds (already done)"
      continue
    fi
    echo -e "\n=== $label  ->  $ds ($SPLIT) ==="
    python -m scripts.predict tiles \
      --dataset_root "$DATASET_ROOT" --dataset_name "$ds" \
      --tile_size "$TILE_DIR" --split "$SPLIT" \
      --model_path "$model" --arch "$ARCH" \
      --batch_size 32 --output_dir "$out"
  done
done

# ---- Summary table ----
python - "$OUT_ROOT" <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for f in sorted(root.glob("*/*/metrics_summary.json")):
    m = json.loads(f.read_text())
    rows.append((f.parent.parent.name, f.parent.name, m))
if not rows:
    sys.exit("No metrics found (no GT in the evaluated split?)")
print(f"\n{'model':<14}{'dataset':<32}{'IoU':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}")
for label, ds, m in rows:
    print(f"{label:<14}{ds:<32}{m['iou']:>8.4f}{m['precision']:>8.4f}{m['recall']:>8.4f}{m['f1']:>8.4f}")
with open(root / "summary.csv", "w") as out:
    out.write("model,dataset,iou,precision,recall,f1\n")
    for label, ds, m in rows:
        out.write(f"{label},{ds},{m['iou']:.4f},{m['precision']:.4f},{m['recall']:.4f},{m['f1']:.4f}\n")
print(f"\nSaved {root / 'summary.csv'}")
EOF