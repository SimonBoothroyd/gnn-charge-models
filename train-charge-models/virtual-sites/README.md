1. Generate the different charge models to train

```shell
python generate-charge-models.py
```

2. Extract a subset of train / test data relevant to the charge model

```shell
MODEL_NAME="vam1bcc-v1"
cd $MODEL_NAME

N_PROCESSES=5

python ../../select-esp-subset.py \
    --input-records   "../../../data-set-labelling/qc-esp/esp-records-fragment-set.pkl" \
    --output-records  "train-esp-records.pkl" \
    --output-coverage "train-coverage.json"   \
    --params-bcc      "v-sites/initial-parameters-bcc.json"    \
    --params-vsite    "v-sites/initial-parameters-v-site.json" \
    --n-processes $N_PROCESSES

python ../../../scripts/records-to-smi.py --input train-esp-records.pkl --output train-esp-records.smi
    
python ../../select-esp-subset.py \
    --input-records   "../../../data-set-labelling/qc-esp/esp-records-industry-set.pkl" \
    --exclusions      "train-esp-records.smi" \
    --output-records  "test-esp-records.pkl" \
    --output-coverage "test-coverage.json"   \
    --params-bcc      "v-sites/initial-parameters-bcc.json"    \
    --params-vsite    "v-sites/initial-parameters-v-site.json" \
    --n-processes 5    
    
python ../../../scripts/records-to-smi.py --input test-esp-records.pkl --output test-esp-records.smi
```

3. Train the model

```shell
bsub < submit-train-test-no-v-sites.sh
bsub < submit-train-test-v-sites.sh
bsub < submit-test-am1bcc.sh
bsub < submit-test-resp.sh
```

4. Plot the output

```shell
python ../../../test-charge-models/plot-test-rmse.py \
    --input "resp"       "test-per-molecule-rmse-resp.json" \
    --input "am1bcc"     "test-per-molecule-rmse-am1bcc.json" \
    --input "no-v-sites" "no-v-sites/lr-0.0025-n-400/test-per-molecule-rmse.json" \
    --input "v-sites"    "v-sites/lr-0.0025-n-800-rad-1.0-str-500-pad-0.3/test-per-molecule-rmse.json"

python ../../../test-charge-models/plot-test-rmse.py \
    --reference "ref"        "test-per-molecule-rmse-am1bcc.json" \
    --input     "no-v-sites" "no-v-sites/lr-0.0025-n-400/test-per-molecule-rmse.json" \
    --input     "v-sites"    "v-sites/lr-0.0025-n-800-rad-1.0-str-500-pad-0.3/test-per-molecule-rmse.json" \
    --output "refit-vs-orig-am1bcc.png"

python ../../../test-charge-models/plot-test-rmse.py \
    --reference "ref"    "no-v-sites/lr-0.0025-n-400/test-per-molecule-rmse.json" \
    --input    "v-sites" "v-sites/lr-0.0025-n-800-rad-1.0-str-500-pad-0.3/test-per-molecule-rmse.json" \
    --output "v-site-vs-no.png"
```
