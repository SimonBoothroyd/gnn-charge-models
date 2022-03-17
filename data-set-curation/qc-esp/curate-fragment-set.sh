#!/bin/bash

mkdir -p data/staging && mkdir -p data/processed

# Generate a set of small, fast to compute, fragments from a diverse set of drug-like
# molecules.
python generate-fragments.py --input  "data/raw/enamine-10240.sdf.gz" \
                             --input  "data/raw/enamine-50240.sdf.gz" \
                             --input  "data/raw/ZINC_eps_78.sdf.gz"   \
                             --input  "data/raw/ChEMBL_eps_78.sdf.gz" \
                             --input  "data/raw/OpenFF-Industry-Benchmark-Season-1-v1-1.smi" \
                             --output "data/staging/fragments"

# Optionally prune the small fragments so that we retain only a diverse subset
#python select-diverse-fragments.py --input  "data/staging/fragments/fragments-small.smi" \
#                                   --output "data/staging/fragments-pruned.smi"          \
#                                   --n-fragments 30000

# Enumerate protomers of the fragments to capture more charge states
nagl prepare enumerate --input  "data/staging/fragments/fragments-small.smi" \
                       --output "data/staging/fragments-enumerated.smi"      \
                       --no-tautomers    \
                       --protomers       \
                       --max-protomers 2 \
                       --n-processes 8

# Join the original and protomer enumerated files to make sure we retain
# ~pH 7 protomers and the original fragments.
python utilities/join-files.py --input  "data/staging/fragments-small.smi"      \
                               --input  "data/staging/fragments-enumerated.smi" \
                               --output "data/processed/esp-fragment-set.smi"
