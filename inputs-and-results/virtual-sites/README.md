1. Generate the different charge models to train

```shell
python generate-charge-models.py
```

2. Extract a subset of train / test data relevant to the charge model

```shell
MODEL_NAME="pyridine-only"
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

3. Train the model

```shell
bsub < submit-train-test-no-v-sites.sh
bsub < submit-train-test-v-sites.sh
```
