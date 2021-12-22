import mlflow.pyfunc


def model_loader(model_name: str):
    '''
    function to load a model from artifacts
    param: model_name
    output: model object
    '''
    model_path = "artifacts/" + model_name
    model = mlflow.pyfunc.load_model(model_path)
    return model


if __name__ == "__main__":

    model_name = "irony_model"
    model = model_loader(model_name)

    print("Note: the model can understand 50+ languages because of the multilingual model that is used for building the embeddings")
    print()
    print(
        "What text do you want to identify? (write it into the terminal)\nif you want to stop, just press enter"
    )
    print()
    answer = str(input())
    print()

    while answer != "":
        prediction = model.predict([answer])
        print("prediction: ", prediction)
        print()
        print(
            "What text do you want to identify? (write it into the terminal)\nif you want to stop, just press enter"
        )
        print()
        answer = str(input())
        print()

    print("--> testing stoped")
