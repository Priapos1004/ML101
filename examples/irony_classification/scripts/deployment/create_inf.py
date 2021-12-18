import mlflow.azureml

from azureml.core import Workspace
from azureml.core.webservice import AciWebservice, Webservice


if __name__ == "__main__":
    # creates a resource group with a ml workspace in it
    workspace_name = "irony-ws"
    subscription_id = "..." # insert your own subscription id
    resource_group = "irony"
    location = "West Europe"
    azure_workspace = Workspace.create(name=workspace_name,
                                       subscription_id=subscription_id,
                                       resource_group=resource_group,
                                       location=location,
                                       create_resource_group=True, # change to False if you already have a resource group
                                       exist_ok=True,
                                      )