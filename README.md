# ML101
This repository is created to help people to get started with Machine Learning and give some tips&amp;tricks for programing with python

**table of content:**

[general workflow](#general_workflow) 

[Step 1: install anaconda](#anaconda) 

[Step 2: virtual environments](#virual_environment)

[Step 3: jupyter notebook](#jupyter_notebook) 

[Step 4: script/.py-files](#scripts) 

[Step 5: some last tips](#tips) 

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

Now that you have a preprocessing for your data and a model with a good performance, you can bring your code in a production ready form. This means to refactor your code into classes, functions and different script. At the end, you want to have a workflow from running script1 --> script2 --> script3 so that you have raw data --> preprocessed data --> train and save model --> deploy model
For example, you could create following scripts:
- data_prep.py --> takes the raw data and returns the preprocessed data
- model.py --> class of model with train and predict function (can contain several models that are called to generate the output)
- train.py --> takes the preprocessed data, trains the model on it and saves the model (e.g.: with the `pickle` library)
- deploy.py --> takes the saved models and deploys them in the cloud (e.g.: azure, AWS, ...)
- consume.py --> takes data, sends them as a request to the deployment endpoint in the cloud and returns the prediction of the model

### (8) implement new features and fix bugs

Your first prototype is now in production, but this is not the end. There are maybe some bugs in the code that have to be fixed or some conflicts with other operating systems. Furthermore, you can start now to find new practical features to implement and create prototype 2/3/... For testing, you can use jupyter notebooks again and afterwards implement the new stuff in your script structure.
- implement new features
- fix bugs/solve problems and conflicts

<a name="anaconda"/>

# Step 1: install anconda

Anaconda is a collection of useful python packages like sklearn or pandas that you will use very often while doing machine learning (So, you can save time with not doing *pip install* for all these packages that you will need). Furthermore, anaconda can be used for managing your virtual environments and running jupyter notebooks.
- [anaconda website](https://www.anaconda.com/products/individual)

<a name="virual_environment"/>

# Step 2: usage of virtual environments

You use the `base` environment when you do commands in the terminal or running py-scripts (if you do not change it). This means that you install (e.g.: with *pip*) all libraries in this environment. That works only until a certain point because the different libraries need different versions of their subpackages and that can produce conflicts. Some of these conflicts can be solved and others not. In the worst case, you cannot use pip anymore and there will be a ton of errors while executing your code. The solution are virtual environments. A virtual environment is a separate environment in which you can install packages and the different environments do not interact with each other. So, you can have for every project a different environment (this brings some advantages that will be mentoined later).

## create a virtual environment

First of all, you have to create an environment

```sh conda create new_env```

Second, you have to activate it

```sh conda activate new_env```

<a name="jupyter_notebook"/>

# Step 3: usage of jupyter notebooks

<a name="scripts"/>

# Step 4: usage of scripts/py-files

<a name="tips"/>

# Step 5: some last tips

## folder structure

How you structure your folders is your choice, but at the end, it has to be understandable for other users which means it should not be too messy. I personally like the following structure:
```
/project name
  /data
    /01_raw
    /02_processed
  /notebooks
    /<name>.ipynb
  /scripts
    /data_prep
      /data_prep.py
      /utils.py (some times if I want to separate the preprocessing function when I need a lot of preprocessing)
    /deployment
      /deploy.py
      /consume.py
      /score.py (script for the scoring endpoint)
    /models
      /model.py (to separate the model class from the train and save script)
      /train_and_save.py
      /utils.py (if I need some functions that make the train_and_save script overcrowded)
  /artifacts
    /model.pkl (saved model)
    /<name>.pkl (if I also need to save an encoder, scaler, ...)
```
The advantage of having the same structure in every project is that others can easily run your projects with always the same workflow. 

(here: data_prep.py --> train_and_save.py --> deploy.py --> consume.py)
