#!/bin/bash
#SBATCH --job-name=tbg_gen
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=0-9%4
#SBATCH --output=logs/tbg_gen_%A_%a.out
#SBATCH --error=logs/tbg_gen_%A_%a.err

set -euo pipefail

# IMPORTANT: Slurm opens --output/--error before this script starts. If this
# is your first run, create the log directory before sbatch:
#   mkdir -p logs
#
# Submit from the repository root.
#
# Use one of these:
#   sbatch scripts/generate_tbg_database.slurm "1.1,2.0,3.5"
#   sbatch --export=ALL,ANGLES="1.1,2.0,3.5" scripts/generate_tbg_database.slurm
#   sbatch --export=ALL,ANGLE_FILE="$PWD/angles.txt" scripts/generate_tbg_database.slurm
#   sbatch --export=ALL,MATERIAL_ID=graphene,TOP_MATERIAL_ID=hbn,ANGLES="2.0" scripts/generate_tbg_database.slurm
#
# If neither ANGLES nor ANGLE_FILE is given, the Python default ANGLE_LIST_DEG is used.
# Adjust --array above to control how many chunks Slurm uses.

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"

mkdir -p logs

if [[ -n "${OUTDIR:-}" ]]; then
    OUTDIR="$OUTDIR"
elif [[ -n "${SCRATCH:-}" ]]; then
    OUTDIR="$SCRATCH/tbg_dataset"
else
    OUTDIR="$PROJECT_DIR/out_tbg_database"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON="$PYTHON_BIN"
elif [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    PYTHON="python"
fi

echo "Project directory: $PROJECT_DIR"
echo "Output directory:  $OUTDIR"
echo "Python:            $PYTHON"
echo "Array task:        ${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}"

"$PYTHON" - <<'PY'
import importlib.util
import sys

missing = [
    name
    for name in ("ase", "numpy", "supercell_core")
    if importlib.util.find_spec(name) is None
]
if missing:
    print(
        "Missing required Python packages: " + ", ".join(missing),
        file=sys.stderr,
    )
    print(
        "Load/activate the environment that has this project installed, or submit with "
        "PYTHON_BIN=/path/to/python.",
        file=sys.stderr,
    )
    sys.exit(2)
PY

ANGLE_ARGS=()
if [[ $# -gt 0 && -n "${1:-}" ]]; then
    ANGLE_ARGS=(--angles "$1")
elif [[ -n "${ANGLES:-}" ]]; then
    ANGLE_ARGS=(--angles "$ANGLES")
elif [[ -n "${ANGLE_FILE:-}" ]]; then
    if [[ ! -f "$ANGLE_FILE" ]]; then
        echo "ANGLE_FILE does not exist on this node: $ANGLE_FILE" >&2
        exit 2
    fi
    ANGLE_ARGS=(--angle-file "$ANGLE_FILE")
fi

MATERIAL_ARGS=(--material-id "${MATERIAL_ID:-graphene}")
if [[ -n "${TOP_MATERIAL_ID:-}" ]]; then
    MATERIAL_ARGS+=(--top-material-id "$TOP_MATERIAL_ID")
fi

PHYSICS_ARGS=()
if [[ -n "${INTERLAYER_DISTANCE:-}" ]]; then
    PHYSICS_ARGS+=(--interlayer-distance "$INTERLAYER_DISTANCE")
fi
if [[ -n "${MAX_EL:-}" ]]; then
    PHYSICS_ARGS+=(--max-el "$MAX_EL")
fi
if [[ -n "${THETA_WINDOW_DEG:-}" ]]; then
    PHYSICS_ARGS+=(--theta-window-deg "$THETA_WINDOW_DEG")
fi
if [[ -n "${THETA_STEP_DEG:-}" ]]; then
    PHYSICS_ARGS+=(--theta-step-deg "$THETA_STEP_DEG")
fi
if [[ -n "${STRAIN_TOL:-}" ]]; then
    PHYSICS_ARGS+=(--strain-tol "$STRAIN_TOL")
fi
if [[ -n "${MAX_ATOMS:-}" ]]; then
    PHYSICS_ARGS+=(--max-atoms "$MAX_ATOMS")
fi

ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"

"$PYTHON" -m tbg_rebuild.cli.generate_tbg_database \
    --outdir "$OUTDIR" \
    "${MATERIAL_ARGS[@]}" \
    "${ANGLE_ARGS[@]}" \
    "${PHYSICS_ARGS[@]}" \
    --array-task-id "$ARRAY_TASK_ID" \
    --array-task-count "$ARRAY_TASK_COUNT" \
    --skip-existing \
    --no-views

if [[ "$ARRAY_TASK_COUNT" -gt 1 ]]; then
    echo "Chunk ledger files are in: $OUTDIR"
    echo "After all array jobs finish, merge with:"
    echo "  cat \"$OUTDIR\"/dataset_results_chunk_*.jsonl > \"$OUTDIR\"/dataset_results.jsonl"
    echo "  cat \"$OUTDIR\"/dataset_failures_chunk_*.jsonl > \"$OUTDIR\"/dataset_failures.jsonl"
else
    echo "Dataset output is in: $OUTDIR"
fi
