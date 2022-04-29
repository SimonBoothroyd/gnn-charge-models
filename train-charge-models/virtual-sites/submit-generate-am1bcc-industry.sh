#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J am1bcc-ind
#BSUB -W 24:00
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
#BSUB -q cpuqueue
#
#BSUB -n 40
#BSUB -M 4

# Enable conda
. ~/.bashrc

conda activate gnn-charge-models
conda env export > "extract-am1-charges.yml"

mkdir reference-charges

python generate-reference-charges-base.py \
    --input  "../../data-set-labelling/qc-esp/esp-records-industry-set.pkl" \
    --output "reference-charges/am1bcc-charge-industry-set.json" \
    --method "am1bcc" \
    --n-processes 40
