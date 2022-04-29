#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J am1bcc
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
conda env export > "test-am1bcc-env.yml"

python ../../../test-charge-models/test-charge-model.py \
    --input-esp-records         "test-esp-records.pkl"      \
    --input-parameters-base     "../../../test-charge-models/reference-charges/am1bcc-charge-industry-set.json"   \
    --output                    "test-per-molecule-rmse-am1bcc.json"  \
    --n-loader-processes        40
