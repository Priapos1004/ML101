from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import mlflow.pyfunc


class irony_model(mlflow.pyfunc.PythonModel):
    
    def __init__():
        '''
        initialise objects that are later needed in training and predicting
        '''
        self.model = LogisticRegression()
        self.labelencoder = LabelEncoder()
        self.CountVectorizer = CountVectorizer(max_features=3000)
        
    def predict(self, data):
        '''
        function to predict if there is irony in a strings
        param: data --> array of strings ["...", "...", ...]
        output: predictions --> array of strings and strings will be "irony" or "regular"
        '''
        
        print("received data: ", data)
        print()
        print("starting predicting...")
        
        # iterate through the data to predict for each text
        results = []
        print("... 0/", len(data), " predictions")
        for i in range(len(data)):
            
            # vectorize text
            text_cv = self.CountVectorizer.transform([data[i]]).toarray()
            
            # predict text
            pred = self.model.predict(text_cv)
            
            # convert prediction (int) to label (str)
            encoded_pred = self.labelencoder.inverse_transform(pred)
            
            # add prediction to result list
            results.append(encoded_pred)
            
            print("... ", (i+1), "/", len(data), " predictions")
        
        print("--> prediction completed")
        
        return results
        
        
    def train(self, X_train_path, y_train_path, X_test_path, y_test_path):
        '''
        function that trains the irony model, the labelencoder for the class column and the vectorizer for the tweets
        param: paths to the processed data
        output: none because the trained models will be saved inside this class
        '''
        
        print("training script started...")
        
        # load processed data
        print("... load data")
        X_train = pd.read_csv(X_train_path)
        y_train = pd.read_csv(y_train_path)
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path)
        
        # encode class labels
        print("... encode class labels")
        y_train["class_encoded"] = self.labelencoder.fit_transform(y_train["class"])
        y_test["class_encoded"] = self.labelencoder.fit_transform(y_test["class"])
        
        # vectorize text
        print("... vectorize tweets")
        X_train_cv = self.CountVectorizer.fit_transform(X_train).toarray()
        X_test_cv = self.CountVectorizer.transform(X_test).toarray()
        
        # training irony model
        print("... start training model")
        self.model.fit(X_train, y_train["class_encoded"])
        
        # give out some metrics
        print("... classificationreport: ")
        pred = lr.predict(X_test)
        print(classification_report(y_test["class_encoded"], pred))
            
        print("--> training script completed")
        