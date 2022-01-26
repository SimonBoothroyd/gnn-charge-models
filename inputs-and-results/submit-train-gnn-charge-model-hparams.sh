#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J gnn-q[1-144]%60
#BSUB -W 03:00
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
# Set any gpu options.
#BSUB -q gpuqueue
#BSUB -gpu num=1:j_exclusive=yes:mode=shared:mps=no:
#
#BSUB -M 5

# Enable conda
. ~/.bashrc

conda activate nagl
conda env export > conda-env-h-params.yml

# Launch my program.
module load cuda/11.0

export batch_size=( 256 )

export n_gcn_layers=( 3 4 5 6 )
export n_gcn_hidden_features=( 32 64 128 )

export n_am1_layers=(0 1 2 3)
export n_am1_hidden_features=( 64 128 256 )

export learning_rate=( 0.001 )

export indices=( $(
  python utilities/job-to-matrix-index.py $LSB_JOBINDEX \
                                          ${#batch_size[@]} \
                                          ${#n_gcn_layers[@]} \
                                          ${#n_gcn_hidden_features[@]} \
                                          ${#n_am1_layers[@]} \
                                          ${#n_am1_hidden_features[@]} \
                                          ${#learning_rate[@]}
) )

echo "MATRIX INDICES=${indices[*]}"

python train-am1-q-model.py --train-set             "../data-set-labelling/am1-charges/chg-enamine-10240.sqlite" \
                            --train-batch-size      ${batch_size[${indices[0]}]} \
                            --val-set               "../data-set-labelling/am1-charges/chg-OpenFF-Industry-Benchmark-Season-1-v1-1.sqlite" \
                            --n-gcn-layers          ${n_gcn_layers[${indices[1]}]} \
                            --n-gcn-hidden-features ${n_gcn_hidden_features[${indices[2]}]} \
                            --n-am1-layers          ${n_am1_layers[${indices[3]}]} \
                            --n-am1-hidden-features ${n_am1_hidden_features[${indices[4]}]} \
                            --learning-rate         ${learning_rate[${indices[5]}]} \
                            --n-epochs              175