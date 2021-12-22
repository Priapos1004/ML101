# irony classification example

**Note:** all commands have to be ran from the *irony_classification folder*

## create virtual environment for the project

To create a virtual environment from `conda.yaml` and activate the environment, run the following commands. The environment has no packages for launching jupyter notebook to keep it minimal for the deployment. Use it for running the scripts. For running the notebooks, use your *experimental environment* (or clone your base environment) and install `sentence-transoformers` additionally (`pip install sentence-transformers´).

```sh
conda env create -f conda.yaml
conda activate irony_env
```

## workflow for creating, training and testing the model

### investigate data and some models in jupyter notebook

run the notebook `irony classification.ipynb`

### preprocess raw data

The script `data_prep.py` will create four .csv-files with the preprocessed data split into train and test that the model can use to train and be evaluated:

```
data/2_processed/X_train.csv
data/2_processed/y_train.csv
data/2_processed/X_test.csv
data/2_processed/y_test.csv
```

To preprocess the raw data, run the following command:

```sh
python scripts/data_prep/data_prep.py
```

### train on preprocessed data

The script `train.py` will create an object of the model class (`model.py`), train it and save it into the *artifacts folder*. To train the model, run the following command:

```sh
python scripts/models/train.py
```

### test trained model

The script `test.py` will allow you to predict text with the model that is given via the terminal (just follow the instructions in the terminal). **Note:** the model can classify 50+ languages because of the multilingual model that is used for creating the embeddings. To test the model, run the following command:

```sh
python scripts/models/test.py
```

## workflow for deploying the model in the cloud (here: azure) [in progress]

**Note:** As a student one can create a free account on azure with 100$ credit.

### building the infrastructure for deployment

To deploy in azure, one needs to create a resource group and a machine learning workspace in it. The `create_inf.py` script does this for you. Just run the following command:

```sh
python scripts/deployment/create_inf.py
```
