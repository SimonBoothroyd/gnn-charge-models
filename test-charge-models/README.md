1. Generate any reference charge models 

```shell
python generate-reference-charges-resp.py

bsub < submit-generate-am1-industry.sh
bsub < submit-generate-am1bcc-industry.sh
```

2. Export the GNN charge model

```shell
python ../train-charge-models/gnn-charge-models/export-model-charges.py \
    --input-records    "../data-set-labelling/qc-esp/esp-records-industry-set.pkl" \
    --input-checkpoint "../train-charge-models/gnn-charge-models/gnn-am1-v2/default/256-3-128-4-128-0.001/checkpoints/epoch=394-step=236210.ckpt" \
    --output           "gnn-charge-models/gnn-am1-v2-base.json" \
    --n-processes 10
```

3. Select a subset of test set ESP data

```shell
python select-test-set.py
```

4. Test the GNN charges

```shell
cd gnn-charge-models

bsub < submit-test-gnn-am1.sh
bsub < submit-test-am1.sh

bsub < submit-test-gnn-am1bcc.sh
bsub < submit-test-am1bcc.sh

bsub < submit-test-resp.sh
```

5. Plot the output

```shell
python ../plot-test-rmse.py \
  --input "am1bcc"          "outputs/test-per-molecule-rmse-am1bcc.json" \
  --input "gnn-am1-v2-bcc"  "outputs/test-per-molecule-rmse-gnn-am1-v2-bcc.json" \
  --input "am1"             "outputs/test-per-molecule-rmse-am1.json" \
  --input "gnn-am1-v2"      "outputs/test-per-molecule-rmse-gnn-am1-v2.json" \
  --input "resp"            "outputs/test-per-molecule-rmse-resp.json"
  
python ../plot-test-rmse.py \
  --reference "am1bcc"      "outputs/test-per-molecule-rmse-am1bcc.json" \
  --input "gnn-am1-v2-bcc"  "outputs/test-per-molecule-rmse-gnn-am1-v2-bcc.json"
  
# Espaloma seems to not yield the correct total charge on charged molecules
python ../plot-test-rmse.py \
  --reference "am1bcc"     "outputs/test-per-molecule-rmse-am1bcc.json" \
  --input "espaloma"       "outputs/test-per-molecule-rmse-espaloma-0-2-2.json" \
  --input "gnn-am1-v2-bcc" "outputs/test-per-molecule-rmse-gnn-am1-v2-bcc.json" \
  --neutral
```

6. Extract the worst performing molecules for closer inspection

```shell
python summarise-results.py
```