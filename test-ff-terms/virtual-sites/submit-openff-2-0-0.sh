#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J fsolv[1]
#BSUB -W 05:59
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
# Set any gpu options.
#BSUB -q gpuqueue
#BSUB -gpu num=1:j_exclusive=yes:mode=shared:mps=no:
#
#BSUB -M 16

# Enable conda
. ~/.bashrc

conda activate gnn-charge-models
conda env export > test-fsolv.yml

# Launch my program.
module load cuda/11.0

python ../run-calculation.py \
  --force-field   "openff-2.0.0.offxml"    \
  --input-systems "test-set.json"          \
  --input-index   $(( $LSB_JOBINDEX - 1 )) \
  --output        "results/openff-2-0-0/$(( $LSB_JOBINDEX - 1 )).json"

