# register model

import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub
from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "adityapaikrao"
repo_name = "mlops"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
# dagshub.init(repo_owner=os.getenv("DAGSHUB_REPO_OWNER"), repo_name=os.getenv("DAGSHUB_REPO_NAME"), mlflow=True)
# -------------------------------------------------------------------------------------


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logging.info(f'Registering model from URI: {model_uri}')
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        logging.info(f'Model {model_name} version {model_version.version} registered successfully.')
        
        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        logging.info(f'Attempting to transition model {model_name} version {model_version.version} to Staging...')
        
        try:
            transition_result = client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging"
            )
            logging.info(f'Successfully transitioned model {model_name} version {model_version.version} to Staging stage.')
            logging.info(f'Transition result: {transition_result}')
        except Exception as transition_error:
            logging.error(f'Failed to transition model to Staging: {transition_error}')
            # Let's try to get the current model version details
            try:
                model_version_details = client.get_model_version(name=model_name, version=model_version.version)
                logging.info(f'Current model version stage: {model_version_details.current_stage}')
            except Exception as detail_error:
                logging.error(f'Could not retrieve model version details: {detail_error}')
            raise transition_error
        
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

