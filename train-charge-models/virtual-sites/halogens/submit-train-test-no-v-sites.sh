#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J no-vsite
#BSUB -W 24:00
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
#BSUB -q cpuqueue
#
#BSUB -n 24
#BSUB -M 4

# Enable conda
. ~/.bashrc

conda activate gnn-charge-models
conda env export > "no-v-sites/train-test-env.yml"

LEARNING_RATE=0.0025
N_EPOCHS=400

OUTPUT_DIR="no-v-sites/lr-$LEARNING_RATE-n-$N_EPOCHS"

python ../train-charge-model.py                                    \
    --input-esp-records         "train-esp-records.pkl"            \
    --input-parameter-coverage  "train-coverage.json"              \
    --input-parameters          "no-v-sites"                       \
    --output-directory          $OUTPUT_DIR                        \
    --learning-rate             $LEARNING_RATE                     \
    --n-epochs                  $N_EPOCHS                          \
    --n-loader-processes        24

python ../../../test-charge-models/test-charge-model.py \
    --input-esp-records         "test-esp-records.pkl"     \
    --input-parameters-base     "$OUTPUT_DIR/final-parameters-base.json"   \
    --input-parameters-bcc      "$OUTPUT_DIR/final-parameters-bcc.json"    \
    --input-parameters-v-site   "$OUTPUT_DIR/final-parameters-v-site.json" \
    --output                    "$OUTPUT_DIR/test-per-molecule-rmse.json"  \
    --n-loader-processes        24
