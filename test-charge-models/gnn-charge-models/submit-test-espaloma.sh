#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J espaloma
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
conda env export > "conda-env.yml"

python ../test-charge-model.py \
    --input-esp-records      "../test-esp-records.pkl"      \
    --input-parameters-base  "../espaloma/espaloma-0-2-2.json"   \
    --output                 "outputs/test-per-molecule-rmse-espaloma-0-2-2.json"  \
    --n-loader-processes     40
