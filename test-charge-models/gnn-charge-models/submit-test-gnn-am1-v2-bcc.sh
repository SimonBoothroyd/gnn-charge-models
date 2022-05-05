#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J gam1bcc
#BSUB -W 24:00
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
#BSUB -q cpuqueue
#
#BSUB -n 40
#BSUB -M 6

# Enable conda
. ~/.bashrc

conda activate gnn-charge-models
conda env export > "test-charges-gnn-am1bcc-env.yml"

mkdir outputs

python ../test-charge-model.py \
    --input-esp-records      "../test-esp-records.pkl"      \
    --input-parameters-base  "gnn-am1-v2.json"   \
    --input-parameters-bcc   "../../train-charge-models/gnn-charge-models/gnn-am1-v2-bcc/lr-0.0025-n-400/final-parameters-bcc.json"  \
    --output                 "outputs/test-per-molecule-rmse-gnn-am1-v2-bcc.json"  \
    --n-loader-processes     40
