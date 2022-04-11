```shell
python generate-parameters.py
```

```shell
cd sp2-nitrogen-only

python ../select-esp-subset.py \
    --input-records   "../../../data-set-labelling/qc-esp/esp-records-fragment-set.pkl" \
    --output-records  "train-esp-records.pkl" \
    --output-coverage "train-coverage.json"   \
    --params-bcc      "v-sites/initial-parameters-bcc.json"    \
    --params-vsite    "v-sites/initial-parameters-v-site.json" \
    --n-processes 10

python ../select-esp-subset.py \
    --input-records   "../../../data-set-labelling/qc-esp/esp-records-industry-set.pkl" \
    --output-records  "test-esp-records.pkl" \
    --output-coverage "test-coverage.json"   \
    --params-bcc      "v-sites/initial-parameters-bcc.json"    \
    --params-vsite    "v-sites/initial-parameters-v-site.json" \
    --n-processes 10
```

```shell
python ../train-charge-model.py                             \
    --input-esp-records         "train-esp-records.pkl"     \
    --input-parameter-coverage  "train-coverage.json"       \
    --input-parameters          "no-v-sites"                \
    --output-directory          "no-v-sites/lr-0.005-n-200" \
    --learning-rate        0.005                            \
    --n-epochs             200                              \
    --n-loader-processes   10

python ../train-charge-model.py                                                     \
    --input-esp-records         "train-esp-records.pkl"                             \
    --input-parameter-coverage  "train-coverage.json"                               \
    --input-parameters          "v-sites"                                           \
    --train-vsite-charge        "[#1,#6:1]~[#7X2:2]~[#1,#6:3]"                      \
                                "DivalentLonePair"                                  \
                                "EP"                                                \
                                1                                                   \
    --train-vsite-coord         "[#1,#6:1]~[#7X2:2]~[#1,#6:3]"                      \
                                "DivalentLonePair"                                  \
                                "EP"                                                \
                                "distance"                                          \
                                1.0 100.0 0.4                                       \
    --output-directory          "v-sites/lr-0.005-n-200-rad-1.0-str-100-wth-0.4"    \
    --learning-rate             0.005                                               \
    --n-epochs                  200                                                 \
    --n-loader-processes        10
```

```shell
python ../test-charge-model.py                              \
    --input-esp-records         "test-esp-records.pkl"      \
    --input-parameters-base     "no-v-sites/lr-0.005-n-200/final-parameters-base.json"  \
    --input-parameters-bcc      "no-v-sites/lr-0.005-n-200/final-parameters-bcc.json"   \
    --output                    "no-v-sites/lr-0.005-n-200/test-per-molecule-rmse.json"

python ../test-charge-model.py                              \
    --input-esp-records         "test-esp-records.pkl"     \
    --input-parameters-base     "v-sites/lr-0.005-n-200-rad-1.0-str-100-wth-0.4/final-parameters-base.json"  \
    --input-parameters-bcc      "v-sites/lr-0.005-n-200-rad-1.0-str-100-wth-0.4/final-parameters-bcc.json"   \
    --input-parameters-vsite    "v-sites/lr-0.005-n-200-rad-1.0-str-100-wth-0.4/final-parameters-vsite.json" \
    --output-directory          "v-sites/lr-0.005-n-200-rad-1.0-str-100-wth-0.4/test-per-molecule-rmse.json"
```