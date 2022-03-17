#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J esp[1-46]
#BSUB -W 12:00
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
#BSUB -q cpuqueue
#
#BSUB -n 8
#BSUB -R "rusage[mem=3/task] span[hosts=1]"

# Enable conda
. ~/.bashrc

conda activate gnn-charge-models
conda env export > "conda-env.yml"

mkdir -p data/processed/esp-fragment-conformers

python generate-conformers.py --input  "data/processed/esp-amino-acid-set.smi"     \
                              --output "data/processed/esp-amino-acid-conformers"  \
                              --n-conformers 5 \
                              --batch-size 5   \
                              --batch-idx $(( $LSB_JOBINDEX - 1 )) \
                              --n-processes 8 \
                              --memory 192  # 3 GB * 8 procs -> GiB
