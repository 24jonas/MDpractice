#!/bin/bash
#SBATCH -J tbg-explore
#SBATCH -o slurm.out
#SBATCH -e slurm.err
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 04:00:00
#SBATCH --mem=0

# Load your environment (example – adapt for your HPRC)
module purge
# module load Anaconda3
# source activate gpaw
# or:
# conda activate gpaw

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

srun -n 1 python scripts/explore_hprc.py \
  --stacks graphene,graphene graphene,hbn,graphene \
  --angles 0:30:3 \
  --mode fast \
  --strain-cap 0.02 \
  --atom-cap 1200 \
  --outdir $SCRATCH/explorer_runs \
  --seed 17 \
  --relax-fmax 0.05 --relax-steps 80 \
  --kmesh-relax 4 4 1 \
  --pw-ecut 500 --kmesh-pw 12 12 1 --bands-npts 240
