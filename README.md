# ML101
This repository is created to help people to get started with Machine Learning and give some tips&amp;tricks for programing with python

**table of content**

[general workflow](#general_workflow) 

<a name="general_workflow"/>

# general workflow with machine learning
```(1) data collection --> (2) data exploration --> (3) data preprocessing --> (4) train model --> (5) evaluate model --> (6) repeat steps 3, 4 and 5 until your model is usable ==> (7) create prototype --> (8) implement more features/fix bugs```

## code in jupyter notebooks
### (1) data collection - the base of your model is the data, so choose wisely

Often, you will get specific data for a project and train your model on it; however, this data can be not enough or very messy and in this case, you can use data with a similar structure (e.g.: from Kaggle) to train your model on it and later finetune it on your specific data.
- [Kaggle.com](https://www.kaggle.com/)
- client data

### (2) data exploration - gain some information about your data

The better you know your dataset, the easier it is for you to understand why your model makes it prediction how it does. Furthermore, if you know the deficits of your data, you can do something against it.
- NaN cells
- duplicates
- balance of target class

### (3) data preprocessing - bring the data in a good shape for your model

After step 2, you know now the deficits of your data and you can do something. Also, you have to convert text to vectors and encode categorical features so that your model can work with them.
- delete/fill NaN cells
- handle duplicates
- upsample or downsample data
- convert text to vectors
- encode/scale/normilize features

### (4) train model - choose a model and train it

There are two main types of models you will probably use: classifier and regressors.
- classifier  (features --> classes)
  - special case: two classes (binary classification) --> there are models specificly for this
  - e.g.: Is a cat, a dog or a horse on a pictures?
- regressor   (features --> values)
  - one can also use regressors for classification --> in some cases this can be helpful
  - e.g.: What is the chance (in percentage) to fail a class based on information about a person?

### (5) evaluate model - use meaningful key figures to evaluate the performance of your model
- look at different metrics like f1-score, recall, precision, ...
- classificationreport and confusionmetrics are helpful to evaluate classifiers

### (6) repeate steps 3, 4, and 5 - improve the metrics of the model so that it is usable
- try different preproceesing --> different encoder, scaler, vectorizer or normilizer
- hypertuning model
- try different models
--> you can do this manually by yourself or use helpful libraries like `TPOT` which will do the upper steps for you (code for the tpot classifier/regressor)

## code in scripts/.py-files
### (7) create prototype

Now that you have a preprocessing for your data and a model with a good performance, you can bring your code in a production ready form. This means to refactor your code into classes, functions and different script.
For example, you could create following scripts:
- data_prep.py --> takes the raw data and returns the preprocessed data
- model.py --> class of model with train and predict function (can contain several models that are called to generate the output)
- train.py --> takes the preprocessed data, trains the model on it and saves the model (e.g.: with the `pickle` library)
- deploy.py --> takes the saved models and deploys them in the cloud (e.g.: azure, AWS, ...)
- consume.py --> takes data, sends them as a request to the deployment endpoint in the cloud and returns the prediction of the model
