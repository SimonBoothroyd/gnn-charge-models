# Inputs and Results

1. Explore different combination of GNN hyper-parameters

```shell
bsub < submit-train-am1-model-hparams.sh
```

2. Train a production model using the best hyper-parameters

```shell
bsub < submit-train-am1-model-prod-v2.sh
```

3. Export the GNN charges as library charges so we can train a set of 
   BCC parameters on top of them

```shell
python export-model-charges.py \
    --input-records     "../../data-set-labelling/qc-esp/esp-records-fragment-set.pkl" \
    --input-checkpoint  "gnn-am1-v2/default/256-3-128-4-128-0.001/checkpoints/epoch=394-step=236210.ckpt" \
    --output            "gnn-am1-v2-bcc/input-parameters/initial-parameters-base.json" \
    --n-processes 10
    
python export-model-charges.py \
    --input-records     "../../data-set-labelling/qc-esp/esp-records-industry-set.pkl" \
    --input-checkpoint  "gnn-am1-v2/default/256-3-128-4-128-0.001/checkpoints/epoch=394-step=236210.ckpt" \
    --output            "../../test-charge-models/gnn-charge-models/gnn-am1-v2.json" \
    --n-processes 10    
```

4. Select the subset of training set ESP records to train BCC parameters to

```shell
# Need to to copy `initial-parameters-bcc.json` from the v-site directory or generate
# from scratch. See e.g. `../virtual-sites/generate-charge-models.py`
python ../select-esp-subset.py \
    --input-records   "../../data-set-labelling/qc-esp/esp-records-fragment-set.pkl" \
    --output-records  "gnn-am1-v2-bcc/train-esp-records.pkl" \
    --output-coverage "gnn-am1-v2-bcc/train-coverage.json"   \
    --params-bcc      "gnn-am1-v2-bcc/input-parameters/initial-parameters-bcc.json"    \
    --params-vsite    "gnn-am1-v2-bcc/input-parameters/initial-parameters-v-site.json" \
    --no-filter-by-v-site \
    --n-processes 10
```

5. Train the BCCs on top of the GNN charges

```shell
cd gnn-am1-v2-bcc
bsub < submit.sh
```

6. See `test-charge-models/gnn-charge-models` for test scripts