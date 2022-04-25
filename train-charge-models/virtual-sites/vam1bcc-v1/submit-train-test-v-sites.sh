#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J vsite
#BSUB -W 24:00
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
#BSUB -q cpuqueue
#
#BSUB -n 40
#BSUB -M 6

# Enable conda
. ~/.bashrc

conda activate gnn-charge-models
conda env export > "v-sites/train-test-env.yml"

LEARNING_RATE=0.0025
N_EPOCHS=800

PENALTY_STRENGTH=500
PENALTY_PADDING=0.3

N_RADII=1.0
N_PENALTY_WIDTH=$(echo "print($N_RADII - 2 * $PENALTY_PADDING)" | python)

CL_RADII=1.75
CL_PENALTY_WIDTH=$(echo "print($CL_RADII - 2 * $PENALTY_PADDING)" | python)

BR_RADII=1.85
BR_PENALTY_WIDTH=$(echo "print($BR_RADII - 2 * $PENALTY_PADDING)" | python)

OUTPUT_DIR="v-sites/lr-$LEARNING_RATE-n-$N_EPOCHS-rad-1.0-str-$PENALTY_STRENGTH-pad-$PENALTY_PADDING"

python ../train-charge-model.py                                    \
    --input-esp-records         "train-esp-records.pkl"            \
    --input-parameter-coverage  "train-coverage.json"              \
    --input-parameters          "v-sites"                          \
    --train-vsite-charge        "[#6X3H1a:2]1:[#7X2a:1]:[#6X3H1a:3]:[#6X3a]:[#6X3a]:[#6X3a]1" \
                                "DivalentLonePair"                 \
                                "EP"                               \
                                0                                  \
    --train-vsite-charge        "[#6A:2]-[#17:1]"                  \
                                "BondCharge"                       \
                                "EP1"                              \
                                0                                  \
    --train-vsite-charge        "[#6a:2]-[#17:1]"                  \
                                "BondCharge"                       \
                                "EP1"                              \
                                0                                  \
    --train-vsite-charge        "[#6A:2]-[#35:1]"                  \
                                "BondCharge"                       \
                                "EP1"                              \
                                0                                  \
    --train-vsite-charge        "[#6a:2]-[#35:1]"                  \
                                "BondCharge"                       \
                                "EP1"                              \
                                0                                  \
    --train-vsite-coord         "[#6X3H1a:2]1:[#7X2a:1]:[#6X3H1a:3]:[#6X3a]:[#6X3a]:[#6X3a]1" \
                                "DivalentLonePair"                       \
                                "EP"                               \
                                "distance"                         \
                                $N_RADII                           \
                                $PENALTY_STRENGTH                  \
                                $N_PENALTY_WIDTH                   \
    --train-vsite-coord         "[#6A:2]-[#17:1]"                  \
                                "BondCharge"                       \
                                "EP1"                              \
                                "distance"                         \
                                $CL_RADII                          \
                                $PENALTY_STRENGTH                  \
                                $CL_PENALTY_WIDTH                  \
    --train-vsite-coord         "[#6a:2]-[#17:1]"                  \
                                "BondCharge"                       \
                                "EP1"                              \
                                "distance"                         \
                                $CL_RADII                          \
                                $PENALTY_STRENGTH                  \
                                $CL_PENALTY_WIDTH                  \
    --train-vsite-coord         "[#6A:2]-[#35:1]"                  \
                                "BondCharge"                       \
                                "EP1"                              \
                                "distance"                         \
                                $BR_RADII                          \
                                $PENALTY_STRENGTH                  \
                                $BR_PENALTY_WIDTH                  \
    --train-vsite-coord         "[#6a:2]-[#35:1]"                  \
                                "BondCharge"                       \
                                "EP1"                              \
                                "distance"                         \
                                $BR_RADII                          \
                                $PENALTY_STRENGTH                  \
                                $BR_PENALTY_WIDTH                  \
    --output-directory          $OUTPUT_DIR                        \
    --learning-rate             $LEARNING_RATE                     \
    --n-epochs                  $N_EPOCHS                          \
    --n-loader-processes        40

python ../test-charge-model.py                              \
    --input-esp-records         "test-esp-records.pkl"     \
    --input-parameters-base     "$OUTPUT_DIR/final-parameters-base.json"   \
    --input-parameters-bcc      "$OUTPUT_DIR/final-parameters-bcc.json"    \
    --input-parameters-v-site   "$OUTPUT_DIR/final-parameters-v-site.json" \
    --output                    "$OUTPUT_DIR/test-per-molecule-rmse.json"  \
    --n-loader-processes        40
