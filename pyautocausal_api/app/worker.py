from celery import Celery
from celery.utils.log import get_task_logger
import pandas as pd
from pathlib import Path
import shutil
import os # For environment variables
import boto3 # Add boto3 import
from botocore.exceptions import NoCredentialsError, ClientError
import time
import json
from datetime import datetime

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

# Configure Celery with appropriate timeouts and settings
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
    accept_content=['json'],
    # Add timeout settings
    task_time_limit=3600,  # Hard time limit (1 hour)
    task_soft_time_limit=3300,  # Soft time limit (55 minutes)
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks to prevent memory leaks
    broker_transport_options={
        'visibility_timeout': 43200,  # 12 hours (in seconds)
    },
    result_expires=86400,  # Results expire after 1 day
)

# Set up enhanced logging
logger = get_task_logger(__name__)

# Configuration from Environment Variables
S3_OUTPUT_BUCKET = os.getenv("S3_OUTPUT_BUCKET")
S3_REGION = os.getenv("AWS_REGION", "us-east-1") # Default region if not set

# Log environment configuration at startup
logger.info(f"Celery worker starting with: CELERY_BROKER_URL={CELERY_BROKER_URL}, S3_OUTPUT_BUCKET={S3_OUTPUT_BUCKET}, AWS_REGION={S3_REGION}")

# Configure boto3 with timeouts and retries
s3_client = boto3.client(
    "s3", 
    region_name=S3_REGION,
    config=boto3.session.Config(
        connect_timeout=5,   # 5 seconds
        read_timeout=300,    # 5 minutes for large files
        retries={'max_attempts': 5}  # More retries for worker operations
    )
)

# Helper function for structured logging
def log_worker_event(event_type, job_id, duration_ms=None, status=None, error=None, **kwargs):
    """Log a structured event with consistent format"""
    log_data = {
        "event": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "job_id": job_id
    }
    
    if duration_ms:
        log_data["duration_ms"] = duration_ms
    if status:
        log_data["status"] = status
    if error:
        log_data["error"] = str(error)
    
    # Add any additional kwargs
    log_data.update(kwargs)
    
    # Log as JSON string for better parsing
    logger.info(json.dumps(log_data))

# Function to parse S3 URI into bucket and path components
def parse_s3_uri(s3_uri):
    """Parse an S3 URI into bucket and key components."""
    if not s3_uri or not s3_uri.startswith("s3://"):
        logger.warning(f"Invalid S3 URI format: {s3_uri}")
        return None, None
    
    # Remove 's3://' prefix and split into bucket and key
    parts = s3_uri[5:].split('/', 1)
    bucket = parts[0]
    # If there's a path component, return it, otherwise return empty string
    key = parts[1] if len(parts) > 1 else ""
    logger.debug(f"Parsed S3 URI: bucket={bucket}, key={key}")
    return bucket, key

# Worker will use a temporary local path for processing
WORKER_TEMP_DIR = Path("/tmp/pyautocausal_worker_space").resolve() # Or another suitable temp location

@celery_app.task(
    bind=True, 
    name='app.worker.run_graph_job',
    # Add retry settings
    autoretry_for=(Exception,),  # Retry for any exception
    retry_kwargs={'max_retries': 3, 'countdown': 5},  # Retry up to 3 times, with 5s between retries
    retry_backoff=True,  # Exponential backoff
    retry_jitter=True,   # Add randomness to prevent thundering herd
)
def run_graph_job(self, job_id: str, input_s3_uri: str, original_filename: str):
    """
    Celery task to download data from S3, run simple_graph, and upload outputs to S3.
    """
    task_start_time = time.time()
    logger.info(f"[{job_id}] Celery task 'run_graph_job' started for input: {input_s3_uri}")
    
    # Configuration validation
    if not S3_OUTPUT_BUCKET:
        logger.error(f"[{job_id}] S3_OUTPUT_BUCKET environment variable is not set. Cannot proceed.")
        raise ValueError("Server configuration error: S3 output bucket not specified.")
    
    # Parse the S3 output bucket URI to get bucket name and base path
    output_bucket_name, output_base_path = parse_s3_uri(S3_OUTPUT_BUCKET)
    if not output_bucket_name:
        logger.error(f"[{job_id}] Invalid S3 URI format for S3_OUTPUT_BUCKET: {S3_OUTPUT_BUCKET}")
        raise ValueError("Server configuration error: Invalid S3 output bucket URI format.")
    
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
    
    log_worker_event(
        event_type="job_processing_started",
        job_id=job_id, 
        input_s3_uri=input_s3_uri,
        output_s3_prefix=f"s3://{output_bucket_name}/{s3_output_key_prefix}"
    )

    try:
        # Create local directories for processing
        setup_start = time.time()
        local_input_file_path.parent.mkdir(parents=True, exist_ok=True)
        local_output_job_path.mkdir(parents=True, exist_ok=True)
        setup_duration = (time.time() - setup_start) * 1000  # ms
        
        log_worker_event(
            event_type="local_dirs_created",
            job_id=job_id,
            duration_ms=setup_duration,
            local_dir=str(local_job_processing_dir)
        )
        
        logger.info(f"[{job_id}] Local processing directory: {local_job_processing_dir}")

        # 1. Download data from S3
        current_meta = {'message': f'Downloading data from {input_s3_uri}.'}
        if not TESTING:
            self.update_state(state='PROCESSING', meta=current_meta)
        logger.info(f"[{job_id}] {current_meta['message']}")
        
        try:
            download_start = time.time()
            s3_bucket, s3_key = parse_s3_uri(input_s3_uri)
            if not s3_bucket or not s3_key:
                raise ValueError(f"Invalid S3 URI format: {input_s3_uri}")
                
            # Check that the file exists before trying to download it
            logger.debug(f"[{job_id}] Checking if S3 object exists: {input_s3_uri}")
            s3_client.head_object(Bucket=s3_bucket, Key=s3_key)
            
            # Download the file with progress logging for large files
            logger.debug(f"[{job_id}] Starting download from {input_s3_uri} to {local_input_file_path}")
            s3_client.download_file(s3_bucket, s3_key, str(local_input_file_path))
            
            download_duration = (time.time() - download_start) * 1000  # ms
            file_size = local_input_file_path.stat().st_size
            
            log_worker_event(
                event_type="s3_download_complete",
                job_id=job_id,
                duration_ms=download_duration,
                file_size_bytes=file_size,
                s3_uri=input_s3_uri
            )
            
            logger.info(f"[{job_id}] Successfully downloaded {input_s3_uri} to {local_input_file_path} ({file_size} bytes, {download_duration:.2f}ms)")
        except NoCredentialsError as e:
            logger.error(f"[{job_id}] AWS credentials not found for S3 download: {e}")
            raise # Propagate to mark task as failed
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            logger.error(f"[{job_id}] S3 download failed for {input_s3_uri}: {error_code} - {e}", exc_info=True)
            if error_code == "404":
                raise FileNotFoundError(f"The S3 object {input_s3_uri} does not exist")
            raise # Propagate
        except Exception as e:
            logger.error(f"[{job_id}] Error during S3 download: {e}", exc_info=True)
            raise


        # 2. Load data (assuming CSV)
        current_meta = {'message': f'Loading data from local copy: {original_filename}.'}
        if not TESTING:
            self.update_state(state='PROCESSING', meta=current_meta)
        logger.info(f"[{job_id}] {current_meta['message']}")
        try:
            csv_start = time.time()
            data = pd.read_csv(local_input_file_path)
            csv_duration = (time.time() - csv_start) * 1000  # ms
            
            log_worker_event(
                event_type="csv_loaded",
                job_id=job_id,
                duration_ms=csv_duration,
                rows=len(data),
                columns=len(data.columns),
                file_name=original_filename
            )
            
            logger.info(f"[{job_id}] Successfully loaded CSV with {len(data)} rows and {len(data.columns)} columns in {csv_duration:.2f}ms")
        except Exception as e:
            logger.error(f"[{job_id}] Error reading input CSV ({local_input_file_path}): {e}", exc_info=True)
            raise ValueError(f"Could not parse input file '{original_filename}' for job {job_id}: {e}")

        # 3. Initialize and run the graph (outputs to local_output_job_path)
        current_meta = {'message': 'Initializing and fitting the causal graph.'}
        if not TESTING:
            self.update_state(state='PROCESSING', meta=current_meta)
        logger.info(f"[{job_id}] {current_meta['message']}")

        try:
            graph_start = time.time()
            
            # Create graph with progress reporting
            graph = simple_graph(output_path=local_output_job_path)
            
            # Log graph initialization
            logger.debug(f"[{job_id}] Graph initialized with output path: {local_output_job_path}")
            
            # Update task state with fitting progress - this is a long-running operation
            current_meta = {'message': 'Fitting causal graph. This may take several minutes for large datasets.'}
            if not TESTING:
                self.update_state(state='PROCESSING', meta=current_meta)
            
            # Log before fitting starts
            logger.info(f"[{job_id}] Starting graph fitting on data with shape {data.shape}")
            
            # Fit the graph
            graph.fit(df=data)
            
            graph_duration = (time.time() - graph_start) * 1000  # ms
            
            log_worker_event(
                event_type="graph_fitting_complete",
                job_id=job_id,
                duration_ms=graph_duration,
                nodes_count=len(graph.nodes) if hasattr(graph, 'nodes') else None
            )
            
            logger.info(f"[{job_id}] Graph fitting complete in {graph_duration:.2f}ms")
        except Exception as e:
            logger.error(f"[{job_id}] Error during graph fitting: {e}", exc_info=True)
            raise ValueError(f"Error during causal graph fitting: {e}")

        # 4. Export Notebook locally
        current_meta = {'message': 'Exporting graph to Jupyter Notebook locally.'}
        if not TESTING:
            self.update_state(state='PROCESSING', meta=current_meta)
        logger.info(f"[{job_id}] {current_meta['message']}")
        try:
            notebook_start = time.time()
            exporter = NotebookExporter(graph)
            notebook_filename = "causal_analysis_notebook.ipynb"
            notebook_full_local_path = local_output_job_path / notebook_filename
            
            logger.debug(f"[{job_id}] Exporting notebook to {notebook_full_local_path}")
            exporter.export_notebook(notebook_full_local_path)
            
            notebook_duration = (time.time() - notebook_start) * 1000  # ms
            notebook_size = notebook_full_local_path.stat().st_size
            
            log_worker_event(
                event_type="notebook_export_complete",
                job_id=job_id,
                duration_ms=notebook_duration,
                notebook_size_bytes=notebook_size
            )
            
            logger.info(f"[{job_id}] Notebook exported locally to {notebook_full_local_path} ({notebook_size} bytes) in {notebook_duration:.2f}ms")
        except Exception as e:
            logger.warning(f"[{job_id}] Failed to export notebook locally: {e}", exc_info=True)
            log_worker_event(
                event_type="notebook_export_failed",
                job_id=job_id,
                error=str(e)
            )
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

        upload_start = time.time()
        uploaded_files_count = 0
        total_uploaded_bytes = 0
        
        # Find all files to upload first
        output_files = list(local_output_job_path.rglob('*'))
        output_files = [f for f in output_files if f.is_file()]
        
        if not output_files:
            logger.warning(f"[{job_id}] No files found in {local_output_job_path} to upload to S3.")
        else:
            logger.info(f"[{job_id}] Found {len(output_files)} files to upload to S3")
        
        # Upload each file with progress tracking
        for item in output_files:
            if item.is_file():
                file_key_in_s3 = s3_output_key_prefix + str(item.relative_to(local_output_job_path))
                file_size = item.stat().st_size
                
                try:
                    logger.debug(f"[{job_id}] Uploading {item.name} ({file_size} bytes) to s3://{output_bucket_name}/{file_key_in_s3}")
                    file_upload_start = time.time()
                    
                    s3_client.upload_file(str(item), output_bucket_name, file_key_in_s3)
                    
                    file_upload_duration = (time.time() - file_upload_start) * 1000  # ms
                    
                    log_worker_event(
                        event_type="file_upload_complete",
                        job_id=job_id,
                        duration_ms=file_upload_duration,
                        file_name=item.name,
                        file_size_bytes=file_size,
                        s3_key=file_key_in_s3
                    )
                    
                    logger.debug(f"[{job_id}] Uploaded {item.name} to s3://{output_bucket_name}/{file_key_in_s3} in {file_upload_duration:.2f}ms")
                    uploaded_files_count += 1
                    total_uploaded_bytes += file_size
                except NoCredentialsError:
                    logger.error(f"[{job_id}] AWS credentials not found for S3 upload of {item.name}.")
                    raise # Critical error
                except ClientError as e:
                    logger.error(f"[{job_id}] S3 upload failed for {item.name}: {e}", exc_info=True)
                    # Decide if one failed upload should fail the whole job
                    # For now, we'll log and continue, but this might need adjustment
        
        upload_duration = (time.time() - upload_start) * 1000  # ms
        
        log_worker_event(
            event_type="s3_uploads_complete",
            job_id=job_id,
            duration_ms=upload_duration,
            files_count=uploaded_files_count,
            total_bytes=total_uploaded_bytes
        )
        
        if uploaded_files_count == 0:
            logger.warning(f"[{job_id}] No files were uploaded to S3.")
        else:
            logger.info(f"[{job_id}] Successfully uploaded {uploaded_files_count} result files ({total_uploaded_bytes} bytes) to S3 in {upload_duration:.2f}ms")


        # Task completed successfully, return the S3 URI for the output "directory"
        output_s3_uri = f"s3://{output_bucket_name}/{s3_output_key_prefix}"
        
        # Record total job time
        total_job_duration = (time.time() - task_start_time) * 1000  # ms
        
        log_worker_event(
            event_type="job_completed_successfully",
            job_id=job_id,
            duration_ms=total_job_duration,
            output_s3_uri=output_s3_uri
        )
        
        logger.info(f"[{job_id}] Task 'run_graph_job' completed successfully in {total_job_duration:.2f}ms. Output at {output_s3_uri}")
        return output_s3_uri

    except Exception as e:
        # Log failure with full details
        error_duration = (time.time() - task_start_time) * 1000  # ms
        logger.error(f"[{job_id}] Task 'run_graph_job' failed after {error_duration:.2f}ms: {e}", exc_info=True)
        
        log_worker_event(
            event_type="job_failed",
            job_id=job_id,
            duration_ms=error_duration,
            error=str(e),
            error_type=type(e).__name__
        )
        
        raise
    finally:
        # Clean up local temporary processing directory
        if local_job_processing_dir.exists():
            try:
                cleanup_start = time.time()
                shutil.rmtree(local_job_processing_dir)
                cleanup_duration = (time.time() - cleanup_start) * 1000  # ms
                
                log_worker_event(
                    event_type="cleanup_complete",
                    job_id=job_id,
                    duration_ms=cleanup_duration,
                    directory=str(local_job_processing_dir)
                )
                
                logger.info(f"[{job_id}] Cleaned up local temporary directory: {local_job_processing_dir} in {cleanup_duration:.2f}ms")
            except Exception as e_clean:
                logger.error(f"[{job_id}] Error cleaning up local temporary directory {local_job_processing_dir}: {e_clean}", exc_info=True)
