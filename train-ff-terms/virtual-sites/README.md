```shell
curl "https://github.com/openforcefield/openff-sage/blob/main/data-set-curation/physical-property/optimizations/data-sets/sage-train-v1.json" \
--output sage-train-v1.json

python ./generate-train-vdw-inputs.py \
--input-training-set      "sage-train-v1.json" \
--input-parameters-base   "../../train-charge-models/virtual-sites/vam1bcc-v1/no-v-sites/lr-0.0025-n-400/final-parameters-base.json"    \
--input-parameters-bcc    "../../train-charge-models/virtual-sites/vam1bcc-v1/no-v-sites/lr-0.0025-n-400/final-parameters-bcc.json"     \
--input-parameters-v-site "../../train-charge-models/virtual-sites/vam1bcc-v1/no-v-sites/lr-0.0025-n-400/final-parameters-v-site.json"  \
--output-dir              "vam1bcc-v1" \
--output-name             "no-v-sites"

python ./generate-train-vdw-inputs.py \
--input-training-set      "sage-train-v1.json" \
--input-parameters-base   "../../train-charge-models/virtual-sites/vam1bcc-v1/v-sites/lr-0.0025-n-800-rad-1.0-str-500-pad-0.3/final-parameters-base.json"   \
 --input-parameters-bcc   "../../train-charge-models/virtual-sites/vam1bcc-v1/v-sites/lr-0.0025-n-800-rad-1.0-str-500-pad-0.3/final-parameters-bcc.json"    \
--input-parameters-v-site "../../train-charge-models/virtual-sites/vam1bcc-v1/v-sites/lr-0.0025-n-800-rad-1.0-str-500-pad-0.3/final-parameters-v-site.json" \
--output-dir              "vam1bcc-v1" \
--output-name             "v-sites"
```