#!/bin/bash
set -euo pipefail

# Run the dataset generator against the three core material targets:
#   1. graphene/graphene (TBG)
#   2. graphene/hBN
#   3. hBN/hBN
#
# Default mode is dry-run, so this is cheap and checks parsing/job planning.
# Set DRY_RUN=0 to generate actual structures.
#
# Examples:
#   bash scripts/test_material_matrix.sh
#   ANGLE_FILE=angles_comprehensive.txt bash scripts/test_material_matrix.sh
#   DRY_RUN=0 ANGLE_FILE=angles_smoke.txt bash scripts/test_material_matrix.sh
#   DRY_RUN=0 OUT_BASE=out_material_matrix_custom ANGLE_FILE=angles_smoke.txt bash scripts/test_material_matrix.sh
#   DRY_RUN=0 MAKE_VIEWS=1 ANGLE_FILE=angles_comprehensive.txt bash scripts/test_material_matrix.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    PYTHON="python"
fi

ANGLE_FILE="${ANGLE_FILE:-angles_comprehensive.txt}"
OUT_BASE="${OUT_BASE:-$PROJECT_DIR/out_material_matrix}"
DRY_RUN="${DRY_RUN:-1}"
MAKE_VIEWS="${MAKE_VIEWS:-0}"
if [[ -z "${MAX_EL:-}" ]]; then
    if [[ "$(basename "$ANGLE_FILE")" == "angles_smoke.txt" ]]; then
        MAX_EL=12
    else
        MAX_EL=80
    fi
fi
THETA_STEP_DEG="${THETA_STEP_DEG:-0.05}"
STRAIN_TOL="${STRAIN_TOL:-5e-4}"

echo "Angle file:        $ANGLE_FILE"
echo "Output base:       $OUT_BASE"
echo "Dry run:           $DRY_RUN"
echo "Make views:        $MAKE_VIEWS"
echo "MAX_EL:            $MAX_EL"
echo "Theta step deg:    $THETA_STEP_DEG"
echo "Strain tol:        $STRAIN_TOL"

run_case() {
    local label="$1"
    local material_id="$2"
    local top_material_id="${3:-}"

    echo
    echo "== $label =="

    args=(
        -m tbg_rebuild.cli.generate_tbg_database
        --angle-file "$ANGLE_FILE"
        --material-id "$material_id"
        --max-el "$MAX_EL"
        --theta-step-deg "$THETA_STEP_DEG"
        --strain-tol "$STRAIN_TOL"
    )

    if [[ "$MAKE_VIEWS" != "1" ]]; then
        args+=(--no-views)
    fi

    if [[ -n "$top_material_id" ]]; then
        args+=(--top-material-id "$top_material_id")
    fi

    if [[ "$DRY_RUN" != "0" ]]; then
        args+=(--dry-run)
    else
        args+=(--outdir "$OUT_BASE/$label" --skip-existing)
    fi

    "$PYTHON" "${args[@]}"
}

run_case "tbg" "graphene"
run_case "graphene_hbn" "graphene" "hbn"
run_case "hbn_hbn" "hbn"

echo
if [[ "$DRY_RUN" != "0" ]]; then
    echo "Material matrix dry-run passed for: tbg, graphene_hbn, hbn_hbn"
else
    echo "Material matrix generation finished under: $OUT_BASE"
fi
