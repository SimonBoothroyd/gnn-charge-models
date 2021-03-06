#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J nagl
#BSUB -W 24:00
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
# Set any cpu options.
#BSUB -n 20 -R "span[ptile=20]"
#BSUB -M 2

# Enable conda
. ~/.bashrc

# Use the right conda environment
conda activate gnn-charge-models

mkdir -p data/staging && mkdir -p data/processed

# Filter the Enamine sets according to the criteria proposed by
# Bleiziffer, Schaller and Riniker (see 10.1021/acs.jcim.7b00663)
for name in "enamine-10240" "enamine-50240"
do

  nagl prepare filter --input  "data/raw/${name}.sdf.gz" \
                      --output "data/staging/chg-${name}.sdf.gz" \
                      --strip-ions \
                      --n-processes 20

done

# We don't need to filter the Rinker sets are they are provided in their
# processed form or the OpenFF data set as this was curated by hand.
for name in "ChEMBL_eps_78.sdf.gz" \
            "ZINC_eps_78.sdf.gz" \
            "OpenFF-Industry-Benchmark-Season-1-v1-1.smi"
do

  cp "data/raw/${name}" "data/staging/chg-${name}"

done

# Enumerate a reasonable set of protomers (~7.4 pH) for each input structure
for name in "enamine-10240.sdf.gz" \
            "enamine-50240.sdf.gz" \
            "ChEMBL_eps_78.sdf.gz" \
            "ZINC_eps_78.sdf.gz" \
            "OpenFF-Industry-Benchmark-Season-1-v1-1.smi"
do

  nagl prepare enumerate --input  "data/staging/chg-${name}" \
                         --output "data/processed/chg-${name%%.*}.sdf.gz" \
                         --no-tautomers \
                         --protomers \
                         --max-protomers 2 \
                         --n-processes 20

done
