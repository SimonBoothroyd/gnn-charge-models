Graph Neural Net Based Charge Models
====================================

This repository contains the scripts, inputs and the results generated as part of the *...* publication.

**Warning:** This repository, its structure and contents, are currently in a state of flux and incompleteness while the 
study is ongoing. We *do not* guarantee the scientific correctness of anything found within, nor do we yet recommend 
using any force field parameters found here.

### Structure

This repository is structured into four main directories:

* `data-set-curation` - contains the script used to curate the train and test data sets.

* `data-set-labelling` - contains the script used to generate labels for the train and test sets.

* `train-charge-models` - contains the input files required to train the new charge models.

* `train-ff-terms` - contains the input files required to train the remaining force field terms using the new charge models.

* `trained-models` - contains the trained models produced by this study. 

### Reproduction

The exact inputs used and outputs reported (including the conda environment used to generate them) in the publication 
have been included as tagged releases to this repository. 

For those looking to reproduce the study, the required dependencies may be obtained directly using conda:

```bash
conda env create --name gnn-charge-models --file environment.yaml
```
