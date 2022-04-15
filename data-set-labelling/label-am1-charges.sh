#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J nagl
#BSUB -W 168:00
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
# Set any cpu options.
#BSUB -n 1 -R "span[ptile=1]"
#BSUB -M 16

# Enable conda
. ~/.bashrc

# Use the right conda environment
conda activate gnn-charge-models
conda env export > label-am1-charges-env.yaml

rm -rf am1-charges && mkdir -p am1-charges

# Compute the AM1 partial charges and multi-conformer WBO for each molecule.
for name in "chg-enamine-10240.sdf.gz" \
            "chg-enamine-50240.sdf.gz" \
            "chg-ChEMBL_eps_78.sdf.gz" \
            "chg-ZINC_eps_78.sdf.gz" \
            "chg-OpenFF-Industry-Benchmark-Season-1-v1-1.sdf.gz"
do

  nagl label --input "../data-set-curation/qc-charges/data/processed/${name}"   \
             --output "am1-charges/${name%%.*}.sqlite"              \
             --conf-rms 0.5                                     \
             --n-workers 300                                    \
             --batch-size 250                                   \
             --worker-type lsf                                  \
             --lsf-memory 4                                     \
             --lsf-walltime "32:00"                             \
             --lsf-queue "cpuqueue"                             \
             --lsf-env "gnn-charge-models"

done