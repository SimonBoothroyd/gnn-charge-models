#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J esp[1]%50
#BSUB -W 24:00
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
#BSUB -q cpuqueue
#
#BSUB -n 6
#BSUB -R "rusage[mem=12/job] span[ptile=6]"

# Enable conda
. ~/.bashrc

conda activate gnn-charge-models
conda env export > "conda-env.yml"

mkdir -p qc-esp/industry-set

export $LSB_JOBINDEX=1

python reconstruct-qc-esp.py --input "qc-esp/esp-industry-records.pkl" \
                             --grid  "grid-settings/fcc-0-7.json"      \
                             --output "qc-esp/industry-set/esp-records-fcc-0-7-$LSB_JOBINDEX.pkl" \
                             --batch-size 32                      \
                             --batch-idx $(( $LSB_JOBINDEX - 1 )) \
                             --n-threads 6   \
                             --memory "12 gb"
