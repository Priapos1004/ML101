# ML101
This repository is created to help people to get started with Machine Learning and give some tips&amp;tricks for programing with python

**table of content:**

[general workflow](#general_workflow) 

[Step 1: install anaconda](#anaconda) 

[Step 2: virtual environments](#virual_environment)

[Step 3: jupyter notebook](#jupyter_notebook) 

[Step 4: script/.py-files](#scripts) 

[Step 5: some last tips](#tips) 

- [folder structure](#folder_structure)
- [TPOT library](#tpot)

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
- look into specific rows (some times it reveals dependencies between columns)

### (3) data preprocessing - bring the data in a good shape for your model

After step 2, you know now the deficits of your data and you can do something. Also, you have to convert text to vectors and encode categorical features so that your model can work with them.
- delete/fill NaN cells
- handle duplicates
- upsample or downsample data
- convert text to vectors
- encode/scale/normilize features
- split dataset into train and test data

### (4) train model - choose a model and train it

There are two main types of models you will probably use: classifier and regressors.
- classifier  (features --> classes)
  - special case: two classes ([binary classification](https://www.learndatasci.com/glossary/binary-classification/)) --> there are models specificly for this
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
--> you can do this manually by yourself or use helpful libraries like [TPOT](#tpot) which will do the upper steps for you (code for the tpot classifier/regressor in the *TPOT library* section)

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

```sh
conda create --name new_env
```

Second, you have to activate it (`(base)` should change to `(new_env)` in your terminal)

```sh
conda activate new_env
```

## save a virtual environment to a .yaml-file

Virtual environments can take quite some storage on your computer that is why you should save an environment to a .yaml-file and delete it when you will not use it in some time. Furthermore, others that will run your projects will not always have to install all the different libraries you used (this can take some time and nerves). So, when you save your environment and put it to the rest of your code, one can just create this environment from the .yaml-file and start working with all the libraries.

For saving the environment, you have to `activate` it first and then run the following command. The file will be saved in your current working directory.

```sh
conda env export > conda.yaml
```

## create a virtual environment from a .yaml-file

The name of the environment is the one inside the .yaml-file.

```sh
conda env create -f conda.yaml
```

## remove a virtual environment and its dependencies

```sh
conda remove --name new_env --all
```

## show a list of all virtual environments

It is good to have an overview of all the environments to see which one are not needed anymore.

```sh
conda info --envs
```

## clone a virtual environment

I recommend not to work in the `base` environment and always to activate a different one. In the `base` environment are all standard libraries installed that one could need (the packages one installed with anaconda) without any conflicts and what I like to do is to clone it. So that you have a new `experimental` environment for example that you can use for testing non-project related stuff. You normally do not do this with project related stuff because the .yaml-files of a project should be minimal.

```sh
conda create --name cloned_new_env --clone new_env
```

<a name="jupyter_notebook"/>

# Step 3: usage of jupyter notebook

Now that we know how to use virtual environments, we can start with notebooks. 

## start jupyter notebook

There are two ways to launch jupyter notebook:

### (1) with the anaconda navigator
- open the `anaconda navigator`
- select the environment you want to use in the upper-left corner (default: `base`)
- click launch jupyter notebook (It will start a localhost)

### (2) with the terminal
- open the `terminal`
- `activate` your environment you want to use
- run in the `terminal` (It will start a localhost)

```sh
jupyter notebook
```

## jupyter notebook nbextensions [recommended]

Jupyter notebook is a nice program, but there are [extensions](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions.html) that can make your life way easier. (Maybe one should first get used to the normal notebooks and the basic shortcuts and so on, but after this one should directly use these)

### installation of nbextensions

I would recommend to create a new virtual environment `exten` for the extensions with cloning the `base` environment because the library tends to have conflicts with other bigger libraries and you can use the extensions in notebooks with other environments (I am not sure if this is true for all the extensions, but for most). You can just not edit the currently activated extensions in other environments. Therefore, you have to start jupyter notebooks with `exten`.

To get these `nbextensions` to know, I recommend to read this [article](https://stephanosterburg.gitbook.io/scrapbook/data-science/jupiter-notebook-tips-and-tricks). It also contains other helpful libraries.

To install the extensions in a new environment copy and run the following commands in the terminal:

```sh
conda create --name exten --clone base
conda activate exten
pip install jupyter_contrib_nbextensions
pip install jupyter_nbextensions_configurator
jupyter contrib nbextension install --user
jupyter nbextensions_configurator enable --user
```

### recommendations for the extensions

Nbextensions has a lot of different extensions and all of them are in a way useful, but to get started with them I would recommend the following ones (I do not list the default ones here):

- `Autopep8` - this extension can `solve simple syntax errors` in your notebook
- `Collapsible Headings` - this extension allows you to `minimize header blocks` which makes it easier to work with big notebooks
- `ExecuteTime` - this extension `times the execution` of each code cell and you do not have to use *%%time*
- `Hinterland` - this extension enables `auto-completion` which makes the programing way faster
- `Initialization cells` - this extension allows you to mark cells as `initialization cells` that means they are ran when you load the notebook. You can, for example, load libraries or datasets you always need directly (more a quality of life upgrade)
- `isort formatter` - this extension can `sort your library imports` alphabetically grouped by module import and so (makes the library imports more readable)
- `Scratchpad` - this extension enables an `expandable cell for quick testing` like current state of a variable (otherwise you always have for the program unnecessary cells that makes the notebook less readable)
- `ScrollDown` - this extension `automatically scrolls down` when you have a long output (quality of life upgrade)
- `Snippets Menu` - this extension is the best of all. It allows you to `save code snippets` in a given format *(file will be inserted into the repo soon)* and `insert them in your code`. This can make your speed-up coding and saves time for searching for the same snippet (e.g.: read from .txt-files) for the thousands time.
- `Split Cells Notebook` - this extension allows you to `put two cells next to each other`. This is useful for comparing graphics or outputs.
- `Table of Contents (2)` - this extension enables you a `table of content` with the *Markdown cell* headers as topics. This is useful for big notebooks

<a name="scripts"/>

# Step 4: usage of scripts/.py-files

There are some things that are good to know about scripts.

## structure

Normally, you have a general structure in your script like:

```
library imports

functions/class

if __name__ == "__main__":
  ... (some code that is executed if the script is dircetly ran. It will not be executed if you import functions/class from this script into another)
```

It is good to start functions and classes with a `docstring` that describes the parameter and output of a function besides the normal comments because this makes it way easier to read your code. There are some [conventions](https://www.python.org/dev/peps/pep-0257/) how to do this. Furthermore, there is also a huge list with other conventions for coding with python besides this which are called [PEP 8](https://www.python.org/dev/peps/pep-0008/). It is not bad to know some of these, but at the end, you do not have to know all of them and, as long as your code is clean, readable and understandable, you should be fine.

## recommended libraries for improving your code

I mentioned in the part before that it is import to have readable code, but you do not have to this all by your own. There are some helpful libraries that will support you.

### pip install isort

`isort` is a library to sort imports alphabetically, and automatically separated into sections and by type. You can run it for one file:
```sh
isort <filename>.py
```
or recursively that means it will run for all files in all subfolder from your current working directory:
```sh
isort .
```

### pip install black

`black` is the uncompromising Python code formatter. It deletes unnecessary whitespaces and formats your lists, dictionaries and similar datatypes. You can run this for one file:
```sh
black <filename>.py
```
or recursively that means it will run for all files in all subfolder from your current working directory:
```sh
black .
```

### pip install flake8

`flake8` is a tool for style guide enforcement. It outputs you the lines in the script that have a unclean style and how to clean them up. You can run this for one file:
```sh
flake8 <filename>.py
```
or recursively that means it will run for all files in all subfolder from your current working directory:
```sh
flake8 .
```

<a name="tips"/>

# Step 5: some last tips

<a name="folder_structure"/>

## folder structure

How you structure your folders is your choice, but at the end, it has to be understandable for other users which means it should not be too messy. I personally like the following structure:
```
/project name
  /data
    /1_raw
    /2_processed
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

<a name="tpot"/>

## TPOT library

The TPOT library will save a lot of work. You just have to give the *tpot classifier* or *tpot regressor* your data and it will automatically try different combinations of preprocessing, models and hypertune the models. The [link](http://epistasislab.github.io/tpot/) to there website where they explain in more detail what they exactly do and to there [github repository](https://github.com/EpistasisLab/tpot).

### installation of tpot

Run the following commands in the terminal:

```sh
pip install deap update_checker tqdm stopit xgboost
pip install tpot
```
### example code for the tpot classifier

```
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
                                    random_state=42, verbosity=2)
pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('tpot_exported_pipeline.py')
```

### example code for the tpot regressor

```
from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

housing = load_boston()
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target,
                                                    train_size=0.75, test_size=0.25, random_state=42)

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_boston_pipeline.py')
```
