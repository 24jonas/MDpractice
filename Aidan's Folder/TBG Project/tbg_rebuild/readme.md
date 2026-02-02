# TBG Rebuild — Part I (Structure Generation)

This repository contains **Part I** of a Twisted Bilayer Graphene (TBG) workflow:

> **Generate, validate, and visualize periodic 2D structures**  
> (tiled or solver-assisted commensurate supercells)  
> **before** running ASE/GPAW relaxations or electronic structure.

Part I **does not run GPAW**.  
It produces clean, validated `Structure` objects and PNG views that are safe to hand off to Part II.

---

## What This Code Does

Given a small Python spec, this code:
1. **Chooses a periodic simulation cell** (tiled or supercell-core commensurate),
2. **Builds the atomic structure**,  
3. **Validates geometry**,  
4. **Optionally applies small in-plane strain**,  
5. **Writes artifacts + a JSONL ledger**.

---

## Supported Modes

### 1. tiled
- AA-stacked monolayers or bilayers
- Simple `(n1, n2)` repeats of the hex primitive
- No twist, no registry search


---

### 2. supercell_core
- Uses supercell-core to find a commensurate bilayer near a target twist angle
- Searches integer supercells within a cutoff (`max_el`)
- Applies solver-reported strain **only if required**
- Flags whether the structure was *corrected* (strained) or not

This is the **production path** for twisted bilayer graphene.

---

## What You Get Out

For each job, the pipeline produces:

out_part1/<job_name>/
├── <job_name>_built_3d_layer.png
├── <job_name>_strained_3d_layer.png (only if strain enabled)
└── results.jsonl (one-line summary per job)

And internally:
- A validated structure object
- A clear record of:
  - atom count,
  - commensurate cell,
  - applied strain (if any),
  - solver metadata.

---

Jobs are defined in tbg_rebuild/cli/run_part1.py using helper constructors:

Example: commensurate TBG via supercell-core
make_spec_tbg_supercell_core(
    job_name="sc_theta_1p1deg",
    material_id="graphene",
    theta_deg=1.1,
    vacuum=24.0,
    interlayer_distance=3.35,
    max_el=20,
    theta_window_deg=0.25,
    theta_step_deg=0.01,
    strain_tol=5e-4,
)

Example: tiled AA bilayer
make_spec_bilayer(
    job_name="aa_tiled_small",
    material_id="graphene",
    repeats=(5, 5),
)

Example Slurm Script
#!/bin/bash
#SBATCH --job-name=TBG_Part1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --output=part1.%j.out
#SBATCH --error=part1.%j.err

cd /scratch/$USER/TBG_Project

module purge
module load Anaconda3/2023.09-0
source activate gpaw

python -m tbg_rebuild.cli.run_part1

## How to Run (Local)

From the project root:

```bash
python -m tbg_rebuild.cli.run_part1

