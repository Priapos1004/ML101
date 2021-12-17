# libraries
import re

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# path to raw data and paths for the preprocessed data
data_path = "data/1_raw/data.csv"
X_train_path = "data/2_processed/X_train.csv"
y_train_path = "data/2_processed/y_train.csv"
X_test_path = "data/2_processed/X_test.csv"
y_test_path = "data/2_processed/y_test.csv"


def clean_text(txt: str):
    """
    param: text in string format
    output: text without hashtags
    """
    txt = str(txt).lower()
    txt = re.sub("#\w+", " ", txt)
    txt = txt.strip()  # removes double whitespaces
    return txt


def preprocess():
    """
    function to preprocess the raw data and save the processed data
    """

    print("preprocessing started...")
    
    # read data
    print("... read data")
    df = pd.read_csv(data_path)

    # deleting NaN rows
    print("... deleting NaN rows")
    print("Length of dataframe before deleting: ", len(df))
    df = df[df["tweets"].notna()]
    df = df[df["class"].notna()]
    print("Length of dataframe after deleting: ", len(df))

    # deleting duplicated rows
    print("... deleting duplicated rows")
    print("Length of dataframe before deleting: ", len(df))
    df = df.drop_duplicates()
    print("Length of dataframe after deleting: ", len(df))

    # clean tweets from hashtags
    print("... clean tweets from hashtags")
    df["tweets_clean"] = df["tweets"].apply(clean_text)

    # splitting data into train and test
    print("... splitting data into test and train")
    X = df["tweets_clean"]
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # convert the arrays back to pd.Dataframes
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    # save processed data (index=False is very important otherwise the old indices will saved as new column)
    print("... save processed data")
    X_train.to_csv(X_train_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    y_test.to_csv(y_test_path, index=False)
    print("--> preprocessing completed")


if __name__ == "__main__":
    # run preprocessing
    preprocess()
