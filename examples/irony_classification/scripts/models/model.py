import mlflow.pyfunc
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


class irony_model(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.language_model = SentenceTransformer("quora-distilbert-multilingual")

    def predict(self, context, data):
        """
        function to predict if there is irony in a strings
        param: data --> array of strings ["...", "...", ...]
        output: predictions --> array of strings and strings will be "irony" or "regular"
        """

        print("received data: ", data)
        print()
        print("starting predicting...")

        # vectorize text
        text_vec = self.build_embeddings(data)

        # predict text
        pred = self.model.predict(text_vec)

        # convert prediction (int) to label (str)
        encoded_pred = list(self.labelencoder.inverse_transform(pred))

        print("--> prediction completed")

        return encoded_pred

    def train(
        self,
        X_train_path,
        y_train_path,
        X_test_path,
        y_test_path,
        X_train_pretrained_path,
        X_test_pretrained_path,
    ):
        """
        function that trains the irony model, the labelencoder for the class column and the vectorizer for the tweets
        param: paths to the processed data
        output: none because the trained models will be saved inside this class
        """

        print("training script started...")

        # load processed data
        print("... load data")
        X_train = pd.read_csv(X_train_path)
        y_train = pd.read_csv(y_train_path)
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path)

        # declare the types of the columns (if unclear, it can cause an error while vectorizing)
        X_train = X_train["tweets_clean"].astype("str")
        X_test = X_test["tweets_clean"].astype("str")

        # encode class labels
        print("... encode class labels")
        self.labelencoder = LabelEncoder()
        y_train["class_encoded"] = self.labelencoder.fit_transform(y_train["class"])
        y_test["class_encoded"] = self.labelencoder.fit_transform(y_test["class"])

        # vectorize text
        print("... vectorize tweets")
        try:
            X_train_pretrained = pd.read_csv(X_train_pretrained_path)
            X_test_pretrained = pd.read_csv(X_test_pretrained_path)
            print(
                "- found already vectorized data. Delete it if you want to create new one"
            )
        except:
            print("- did not find already vectorized data --> create new one")
            X_train_pretrained = self.build_embeddings(X_train)
            X_test_pretrained = self.build_embeddings(X_test)
            X_train_pretrained.to_csv(X_train_pretrained_path, index=False)
            X_test_pretrained.to_csv(X_test_pretrained_path, index=False)
            print("- vectorized data created and saved")

        # training irony model
        print("... start training model")
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X_train_pretrained, y_train["class_encoded"])

        # give out some metrics
        print("... classificationreport: ")
        pred = self.model.predict(X_test_pretrained)
        print(classification_report(y_test["class_encoded"], pred))

        print("--> training script completed")

    def build_embeddings(self, data):

        # Embedding creation
        print("- creating embeddings")
        message_embeddings = [self.language_model.encode(str(i)) for i in tqdm(data)]
        ar = np.asarray(message_embeddings)
        df_BERT = pd.DataFrame(ar)
        print("- embeddings created")

        return df_BERT
