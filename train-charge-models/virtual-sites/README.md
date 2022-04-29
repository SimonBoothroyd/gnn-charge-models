1. Generate any reference charge models 

```shell
python generate-reference-charges-resp.py

bsub < submit-generate-am1-industry.sh
bsub < submit-generate-am1bcc-industry.sh
```

2. Generate the different charge models to train

```shell
python generate-charge-models.py
```

3. Extract a subset of train / test data relevant to the charge model

```shell
MODEL_NAME="vam1bcc-v1"
cd $MODEL_NAME

N_PROCESSES=5

python ../select-train-test-subset.py \
    --input-records   "../../../data-set-labelling/qc-esp/esp-records-fragment-set.pkl" \
    --output-records  "train-esp-records.pkl" \
    --output-coverage "train-coverage.json"   \
    --params-bcc      "v-sites/initial-parameters-bcc.json"    \
    --params-vsite    "v-sites/initial-parameters-v-site.json" \
    --n-processes $N_PROCESSES

python ../../../scripts/records-to-smi.py --input train-esp-records.pkl --output train-esp-records.smi
    
python ../select-train-test-subset.py \
    --input-records   "../../../data-set-labelling/qc-esp/esp-records-industry-set.pkl" \
    --exclusions      "train-esp-records.smi" \
    --output-records  "test-esp-records.pkl" \
    --output-coverage "test-coverage.json"   \
    --params-bcc      "v-sites/initial-parameters-bcc.json"    \
    --params-vsite    "v-sites/initial-parameters-v-site.json" \
    --n-processes 5    
    
python ../../../scripts/records-to-smi.py --input test-esp-records.pkl --output test-esp-records.smi
```

4. Train the model

```shell
bsub < submit-train-test-no-v-sites.sh
bsub < submit-train-test-v-sites.sh
bsub < submit-test-am1bcc.sh
bsub < submit-test-resp.sh
```

5. Plot the output

```shell
python ../plot-test-rmse.py \
    --input "resp"       "test-per-molecule-rmse-resp.json" \
    --input "am1bcc"     "test-per-molecule-rmse-am1bcc.json" \
    --input "no-v-sites" "no-v-sites/lr-0.0025-n-400/test-per-molecule-rmse.json" \
    --input "v-sites"    "v-sites/lr-0.0025-n-800-rad-1.0-str-500-pad-0.3/test-per-molecule-rmse.json"

python ../plot-test-rmse.py \
    --reference "ref"        "test-per-molecule-rmse-am1bcc.json" \
    --input     "no-v-sites" "no-v-sites/lr-0.0025-n-400/test-per-molecule-rmse.json" \
    --input     "v-sites"    "v-sites/lr-0.0025-n-800-rad-1.0-str-500-pad-0.3/test-per-molecule-rmse.json" \
    --output "refit-vs-orig-am1bcc.png"

python ../plot-test-rmse.py \
    --reference "ref"    "no-v-sites/lr-0.0025-n-400/test-per-molecule-rmse.json" \
    --input    "v-sites" "v-sites/lr-0.0025-n-800-rad-1.0-str-500-pad-0.3/test-per-molecule-rmse.json" \
    --output "v-site-vs-no.png"    
```

