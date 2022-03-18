#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J esp[1-1250]%50
#BSUB -W 24:00
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
#BSUB -q cpuqueue
#
#BSUB -n 8
#BSUB -R "rusage[mem=6/job] span[ptile=8]"

# Enable conda
. ~/.bashrc

conda activate gnn-charge-models
conda env export > "qc-esp/industry-set-conda-env.yml"

python reconstruct-qc-esp.py --input  "qc-esp/esp-industry-records.pkl" \
                             --output "qc-esp/industry-set/fcc-default-$LSB_JOBINDEX.pkl" \
                             --grid   "grid-settings/fcc-default.json"  \
                             --batch-size 32                            \
                             --batch-idx $(( $LSB_JOBINDEX - 1 ))       \
                             --n-threads 6                              \
                             --memory "6 gb"
