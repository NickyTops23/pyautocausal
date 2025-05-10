from celery import Celery
from celery.utils.log import get_task_logger
import pandas as pd
from pathlib import Path
import shutil
import os # For environment variables
import boto3 # Add boto3 import
from botocore.exceptions import NoCredentialsError, ClientError

# --- pyautocausal Import ---
# This should work if pyautocausal is installed via poetry from the local path
from pyautocausal.pipelines.example_graph import simple_graph
from pyautocausal.persistence.notebook_export import NotebookExporter
# If the above import fails, double-check:
# 1. You ran `poetry install` in the `pyautocausal_api` directory.
# 2. The path `../pyautocausal` in `pyautocausal_api/pyproject.toml` is correct.
# 3. Your `pyautocausal` library has a proper structure (e.g., `pyautocausal/pyautocausal/__init__.py`
#    and `pyautocausal/pyautocausal/pipelines/example_graph.py`).

# Testing flag to disable update_state calls during tests
TESTING = os.getenv('TESTING', 'False').lower() == 'true'

# Initialize Celery
# The first argument 'tasks' is the conventional name for the main module of tasks.
# We'll refer to this celery_app instance when running the worker.
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

celery_app = Celery(
    'tasks', # This is the name of the current module for Celery's purposes
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    # Set the include to ensure tasks are found if worker.py is not the main module
    # when celery worker is started. For simple cases like this, it might not be strictly
    # necessary if you start celery with `-A app.worker.celery_app`
    include=['app.worker']
)

celery_app.conf.update(
    task_track_started=True,
    # It's good practice to use JSON for serialization if possible
    task_serializer='json',
    result_serializer='json',
    accept_content=['json']
)

logger = get_task_logger(__name__)

# Configuration from Environment Variables
S3_OUTPUT_BUCKET = os.getenv("S3_OUTPUT_BUCKET")
S3_REGION = os.getenv("AWS_REGION", "us-east-1") # Default region if not set

s3_client = boto3.client("s3", region_name=S3_REGION)

# Function to parse S3 URI into bucket and path components
def parse_s3_uri(s3_uri):
    """Parse an S3 URI into bucket and key components."""
    if not s3_uri or not s3_uri.startswith("s3://"):
        return None, None
    
    # Remove 's3://' prefix and split into bucket and key
    parts = s3_uri[5:].split('/', 1)
    bucket = parts[0]
    # If there's a path component, return it, otherwise return empty string
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key

# Worker will use a temporary local path for processing
WORKER_TEMP_DIR = Path("/tmp/pyautocausal_worker_space").resolve() # Or another suitable temp location

# Remove local BASE_INPUT_PATH and BASE_OUTPUT_PATH, as we use S3 and temp dirs
# BASE_INPUT_PATH = Path("./local_job_files/inputs").resolve()
# BASE_OUTPUT_PATH = Path("./local_job_files/outputs").resolve()

@celery_app.task(bind=True, name='app.worker.run_graph_job')
def run_graph_job(self, job_id: str, input_s3_uri: str, original_filename: str):
    """
    Celery task to download data from S3, run simple_graph, and upload outputs to S3.
    """
    if not S3_OUTPUT_BUCKET:
        logger.error(f"[{job_id}] S3_OUTPUT_BUCKET environment variable is not set. Cannot proceed.")
        raise ValueError("Server configuration error: S3 output bucket not specified.")
    
    # Parse the S3 output bucket URI to get bucket name and base path
    output_bucket_name, output_base_path = parse_s3_uri(S3_OUTPUT_BUCKET)
    if not output_bucket_name:
        logger.error(f"[{job_id}] Invalid S3 URI format for S3_OUTPUT_BUCKET: {S3_OUTPUT_BUCKET}")
        raise ValueError("Server configuration error: Invalid S3 output bucket URI format.")

    logger.info(f"[{job_id}] Celery task 'run_graph_job' started for input: {input_s3_uri}")

    # Define local paths for this job on the worker instance
    local_job_processing_dir = WORKER_TEMP_DIR / job_id
    local_input_file_path = local_job_processing_dir / "input" / original_filename
    local_output_job_path = local_job_processing_dir / "output"

    # Construct the full S3 key including the base path if it exists
    if output_base_path:
        # Ensure base_path has trailing slash but not leading slash
        output_base_path = output_base_path.strip('/')
        if output_base_path:
            output_base_path += '/'
        s3_output_key_prefix = f"{output_base_path}outputs/{job_id}/"
    else:
        s3_output_key_prefix = f"outputs/{job_id}/"

    try:
        # Create local directories for processing
        local_input_file_path.parent.mkdir(parents=True, exist_ok=True)
        local_output_job_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"[{job_id}] Local processing directory: {local_job_processing_dir}")

        # 1. Download data from S3
        current_meta = {'message': f'Downloading data from {input_s3_uri}.'}
        if not TESTING:
            self.update_state(state='PROCESSING', meta=current_meta)
        logger.info(f"[{job_id}] {current_meta['message']}")
        try:
            s3_bucket, s3_key = input_s3_uri.replace("s3://", "").split("/", 1)
            s3_client.download_file(s3_bucket, s3_key, str(local_input_file_path))
            logger.info(f"[{job_id}] Successfully downloaded {input_s3_uri} to {local_input_file_path}")
        except NoCredentialsError:
            logger.error(f"[{job_id}] AWS credentials not found for S3 download.")
            raise # Propagate to mark task as failed
        except ClientError as e:
            logger.error(f"[{job_id}] S3 download failed for {input_s3_uri}: {e}", exc_info=True)
            raise # Propagate
        except Exception as e:
            logger.error(f"[{job_id}] Error during S3 download setup: {e}", exc_info=True)
            raise


        # 2. Load data (assuming CSV)
        current_meta = {'message': f'Loading data from local copy: {original_filename}.'}
        if not TESTING:
            self.update_state(state='PROCESSING', meta=current_meta)
        logger.info(f"[{job_id}] {current_meta['message']}")
        try:
            data = pd.read_csv(local_input_file_path)
        except Exception as e:
            logger.error(f"[{job_id}] Error reading input CSV ({local_input_file_path}): {e}", exc_info=True)
            raise ValueError(f"Could not parse input file '{original_filename}' for job {job_id}: {e}")

        # 3. Initialize and run the graph (outputs to local_output_job_path)
        current_meta = {'message': 'Initializing and fitting the causal graph.'}
        if not TESTING:
            self.update_state(state='PROCESSING', meta=current_meta)
        logger.info(f"[{job_id}] {current_meta['message']}")

        graph = simple_graph(output_path=local_output_job_path)
        graph.fit(df=data)
        logger.info(f"[{job_id}] Graph fitting complete. Local results in {local_output_job_path}")

        # 4. Export Notebook locally
        current_meta = {'message': 'Exporting graph to Jupyter Notebook locally.'}
        if not TESTING:
            self.update_state(state='PROCESSING', meta=current_meta)
        logger.info(f"[{job_id}] {current_meta['message']}")
        try:
            exporter = NotebookExporter(graph)
            notebook_filename = "causal_analysis_notebook.ipynb"
            notebook_full_local_path = local_output_job_path / notebook_filename
            exporter.export_notebook(notebook_full_local_path)
            logger.info(f"[{job_id}] Notebook exported locally to {notebook_full_local_path}")
        except Exception as e:
            logger.warning(f"[{job_id}] Failed to export notebook locally: {e}", exc_info=True)
            # Non-critical failure for notebook export, proceed to upload other results

        # 5. Upload all contents of local_output_job_path to S3
        current_meta = {'message': 'Uploading results to S3.'}
        if not TESTING:
            self.update_state(state='PROCESSING', meta=current_meta)
        logger.info(f"[{job_id}] Uploading results from {local_output_job_path} to S3 bucket {output_bucket_name} with prefix {s3_output_key_prefix}")

        # For testing, ensure there's at least one file to upload
        if TESTING and local_output_job_path.exists():
            debug_file_path = local_output_job_path / "debug_info.txt"
            with open(debug_file_path, 'w') as f:
                f.write(f"Debug info for job {job_id} - Testing mode")
            logger.info(f"[{job_id}] Created test debug file at {debug_file_path}")

        uploaded_files_count = 0
        for item in local_output_job_path.rglob('*'): # rglob to get all files in subdirectories too
            if item.is_file():
                file_key_in_s3 = s3_output_key_prefix + str(item.relative_to(local_output_job_path))
                try:
                    s3_client.upload_file(str(item), output_bucket_name, file_key_in_s3)
                    logger.info(f"[{job_id}] Uploaded {item.name} to s3://{output_bucket_name}/{file_key_in_s3}")
                    uploaded_files_count += 1
                except NoCredentialsError:
                    logger.error(f"[{job_id}] AWS credentials not found for S3 upload of {item.name}.")
                    raise # Critical error
                except ClientError as e:
                    logger.error(f"[{job_id}] S3 upload failed for {item.name}: {e}", exc_info=True)
                    # Decide if one failed upload should fail the whole job
                    # For now, we'll log and continue, but this might need adjustment
        
        if uploaded_files_count == 0:
            logger.warning(f"[{job_id}] No files were found in {local_output_job_path} to upload to S3.")
        else:
            logger.info(f"[{job_id}] Successfully uploaded {uploaded_files_count} result files to S3.")


        # Task completed successfully, return the S3 URI for the output "directory"
        output_s3_uri = f"s3://{output_bucket_name}/{s3_output_key_prefix}"
        return output_s3_uri

    except Exception as e:
        logger.error(f"[{job_id}] Task 'run_graph_job' failed: {e}", exc_info=True)
        raise
    finally:
        # Clean up local temporary processing directory
        if local_job_processing_dir.exists():
            try:
                shutil.rmtree(local_job_processing_dir)
                logger.info(f"[{job_id}] Cleaned up local temporary directory: {local_job_processing_dir}")
            except Exception as e_clean:
                logger.error(f"[{job_id}] Error cleaning up local temporary directory {local_job_processing_dir}: {e_clean}", exc_info=True)
