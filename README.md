# ML101
This repository is created to help people to get started with Machine Learning and give some tips&amp;tricks for programing with python


# general workflow with machine learning
(1) data collection --> (2) data exploration --> (3) data preprocessing --> (4) train model --> (5) evaluate model --> (6) repeat steps 3, 4 and 5 until your model is usable ==> (7) create Prototype --> (8) implement more features/fix bugs

## in jupyter notebooks
(1) data collection
- Kaggle.com
- client data

(2) data exploration - gain some information about your data
- NaN cells
- duplicates
- balance of target class

(3) data preprocessing
- delete/fill NaN cells
- handle duplicates
- upsample or downsample data
- convert text to vectors
- encode/scale/normilize features

(4) train model
- classifier  (features --> classes)
  - special case: two classes (binary classification) --> there are models specificly for this
  - e.g.: Is a cat, a dog or a horse on a pictures?
- regressor   (features --> values)
  - one can also use regressors for classification --> in some cases this can be helpful
  - e.g.: What is the chance (in percentage) to fail a class based on information about a person?
