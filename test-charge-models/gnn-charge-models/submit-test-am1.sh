#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J am1
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
conda env export > "test-charges-am1-env.yml"

mkdir outputs

python ../test-charge-model.py \
    --input-esp-records      "../test-esp-records.pkl"      \
    --input-parameters-base  "../reference-charges/am1-charge-industry-set.json"   \
    --output                 "outputs/test-per-molecule-rmse-am1.json"  \
    --n-loader-processes     40
