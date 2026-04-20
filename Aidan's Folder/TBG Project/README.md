# TBG Dataset Generator

Generate validated Twisted Bilayer Graphene (TBG) structures from a list of
twist angles.

This project is meant for local smoke tests and HPRC/Slurm dataset runs. It
does **not** run ASE/GPAW relaxation.

---

## What This Code Does

Given one or more twist angles, the generator:

1. Finds a commensurate TBG supercell with `supercell-core`,
2. Builds the full bilayer structure,
3. Validates the geometry,
4. Writes an `.extxyz` structure file,
5. Writes JSONL metadata for the run.

The main production entrypoint is:

```bash
python -m tbg_rebuild.cli.generate_tbg_database
```

---

## Install

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


---

## Local Test

Run a small local example:

```bash
.venv/bin/python -m tbg_rebuild.cli.generate_tbg_database \
  --angles 2.0 \
  --outdir /tmp/tbg_test \
  --no-views
```

To make diagnostic PNG views, omit `--no-views`.

---

## Angle Input

Pass angles directly:

```bash
.venv/bin/python -m tbg_rebuild.cli.generate_tbg_database --angles 1.1,2.0,3.5
```

Or use an angle file:

```bash
.venv/bin/python -m tbg_rebuild.cli.generate_tbg_database --angle-file angles.txt
```

Example `angles.txt`:

```text
1.1
2.0
3.5
29.84
```

Commas and comments are allowed:

```text
1.1, 2.0
3.5   # test angle
```

---

## Slurm

Submit from the project root:

```bash
sbatch scripts/generate_tbg_database.slurm "1.1,2.0,3.5"
```

Or submit an angle file:

```bash
sbatch --export=ALL,ANGLE_FILE="$PWD/angles.txt" scripts/generate_tbg_database.slurm
```

The Slurm script uses a job array:

```bash
#SBATCH --array=0-9%4
```

This means:

- split the angle list into 10 chunks,
- run at most 4 chunks at once.

---

## Outputs

By default, output goes to:

```text
$SCRATCH/tbg_dataset/
```

Local fallback output goes to:

```text
out_tbg_database/
```

Typical output:

```text
tbg_dataset/
├── structures/
│   ├── tbg_theta_1p1deg.extxyz
│   └── tbg_theta_2deg.extxyz
├── artifacts/
│   └── ... optional PNG views ...
├── dataset_results_chunk_0000_of_0010.jsonl
└── dataset_failures_chunk_0000_of_0010.jsonl
```

After a Slurm array run, merge ledgers with:

```bash
cat "$SCRATCH"/tbg_dataset/dataset_results_chunk_*.jsonl > "$SCRATCH"/tbg_dataset/dataset_results.jsonl
cat "$SCRATCH"/tbg_dataset/dataset_failures_chunk_*.jsonl > "$SCRATCH"/tbg_dataset/dataset_failures.jsonl
```

---

## Metadata

Each JSONL record includes:

- `theta_deg`: requested twist angle,
- `n_atoms`: generated atom count,
- `corrected`: whether `supercell-core` applied strain above tolerance,
- `max_strain`: solver-reported strain,
- `strain_tol`: threshold used for `corrected`,
- `validation_built_ok`: whether geometry validation passed,
- `structure_path`: generated `.extxyz` file.

---

## Useful Commands

Dry run:

```bash
.venv/bin/python -m tbg_rebuild.cli.generate_tbg_database --dry-run --angles 1.1,2.0
```

Run one chunk locally:

```bash
.venv/bin/python -m tbg_rebuild.cli.generate_tbg_database \
  --angles 1.1,2.0,3.5 \
  --chunk-count 2 \
  --chunk-index 0
```

Resume without overwriting existing structures:

```bash
.venv/bin/python -m tbg_rebuild.cli.generate_tbg_database \
  --angle-file angles.txt \
  --skip-existing
```
