#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1
#SBATCH --job-name ben_ablation_initial
#SBATCH --partition interactive
#SBATCH --output=ben_ablation.out
#SBATCH --error=ben_ablation.err

export PYTHONPATH=/data/ben_lerner/spar-2024:$PYTHONPATH
srun python experiments/ablations/ablate_largest_correlations.py --first-layer 0 --last-layer 11 --nb 64 --bs 32 --num-feats 100 --no-dry-run
