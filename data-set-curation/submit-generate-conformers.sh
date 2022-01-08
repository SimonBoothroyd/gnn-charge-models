#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J esp[1-440]%40
#BSUB -W 24:00
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
#BSUB -q cpuqueue
#
#BSUB -n 8
#BSUB -R "rusage[mem=4/task] span[hosts=1]"

# Enable conda
. ~/.bashrc

conda activate gnn-charge-models
conda env export > "conda-env.yml"

mkdir -p esp-conformers

python generate-conformers.py --input "processed/esp-molecule-set.smi" \
                              --output "esp-conformers" \
                              --n-conformers 5 \
                              --batch-size 64 \
                              --batch-idx $(( $LSB_JOBINDEX - 1 )) \
                              --n-processes 8 \
                              --memory 238  # 4 GB * 8 procs -> GiB
