#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J esp[1-1550]%300
#BSUB -W 24:00
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
#BSUB -q cpuqueue
#
#BSUB -n 1
#BSUB -R "rusage[mem=4/task] span[hosts=1]"

# Enable conda
. ~/.bashrc

conda activate gnn-charge-models
conda env export > "conda-env.yml"

python label-resp-charges.py --input "resp-fragment-charges/esp-fragment-records.pkl" \
                             --output "resp-fragment-charges/resp-charges" \
                             --batch-size 32 \
                             --batch-idx $(( $LSB_JOBINDEX - 1 ))
