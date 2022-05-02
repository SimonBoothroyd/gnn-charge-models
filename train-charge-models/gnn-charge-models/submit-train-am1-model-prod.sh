#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J gnn-q
#BSUB -W 05:59
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
# Set any gpu options.
#BSUB -q gpuqueue
#BSUB -gpu num=1:j_exclusive=yes:mode=shared:mps=no:
#
#BSUB -M 16

# Enable conda
. ~/.bashrc

conda activate gnn-charge-models
conda env export > train-gnn-charge-model-prod.yml

# Launch my program.
module load cuda/11.0

python train-gnn-charge-model.py --train-set             "../../data-set-labelling/am1-charges/chg-ChEMBL_eps_78.sqlite" \
                                 --train-set             "../../data-set-labelling/am1-charges/chg-ZINC_eps_78.sqlite"   \
                                 --train-batch-size      256                                                             \
                                 --val-set               "../../data-set-labelling/am1-charges/chg-enamine-10240.sqlite" \
                                 --test-set              "../../data-set-labelling/am1-charges/chg-OpenFF-Industry-Benchmark-Season-1-v1-1.sqlite" \
                                 --charge-method         "am1"                                                        \
                                 --n-gcn-layers          3                                                            \
                                 --n-gcn-hidden-features 128                                                          \
                                 --n-am1-layers          4                                                            \
                                 --n-am1-hidden-features 128                                                          \
                                 --learning-rate         0.001                                                        \
                                 --n-epochs              400                                                          \
                                 --output-dir            "gnn-am1"
