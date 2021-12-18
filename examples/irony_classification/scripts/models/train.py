import mlflow.pyfunc
from model import irony_model

X_train_path = "data/2_processed/X_train.csv"
y_train_path = "data/2_processed/y_train.csv"
X_test_path = "data/2_processed/X_test.csv"
y_test_path = "data/2_processed/y_test.csv"

X_train_pretrained_path = "data/2_processed/X_train_pretrained.csv"
X_test_pretrained_path = "data/2_processed/X_test_pretrained.csv"


def train_and_save(model_name: str):

    model = irony_model()
    model.train(
        X_train_path,
        y_train_path,
        X_test_path,
        y_test_path,
        X_train_pretrained_path,
        X_test_pretrained_path,
    )

    mlflow.pyfunc.save_model(
        "artifacts/" + model_name, conda_env="conda.yaml", python_model=model
    )  # code_path=["bbg_quality_return"],


if __name__ == "__main__":
    # name of irony model
    name = "irony_model"

    # run train_and_save
    train_and_save(name)
