#!/bin/bash

mkdir -p staging && mkdir -p processed

# Generate a set of small, fast to compute, fragments from a diverse set of drug-like
# molecules.
python generate-fragments.py --input "raw/enamine-10240.sdf.gz" \
                             --input "raw/enamine-50240.sdf.gz" \
                             --input "raw/ZINC_eps_78.sdf.gz" \
                             --input "raw/ChEMBL_eps_78.sdf.gz" \
                             --input "raw/OpenFF-Industry-Benchmark-Season-1-v1-1.smi" \
                             --output staging/fragments

# Prune the small fragments so that we retain only a diverse subset
#python prune-fragments.py --input "staging/fragments/fragments-small.smi" \
#                          --output "staging/fragments-pruned.smi" \
#                          --n-fragments 30000

# Enumerate protomers of the fragments to capture more charge states
nagl prepare enumerate --input "staging/fragments/fragments-small.smi" \
                       --output "staging/fragments-enumerated.smi" \
                       --no-tautomers \
                       --protomers \
                       --max-protomers 2 \
                       --n-processes 8

# Join the original and protomer enumerated files to make sure we retain
# ~pH 7 protomers and the original fragments.
python utilities/join-files.py --input "staging/fragments-pruned.smi" \
                               --input "staging/fragments-enumerated.smi" \
                               --output "processed/esp-fragment-set.smi"
