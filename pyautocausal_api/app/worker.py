import pandas as pd
from pathlib import Path
import shutil
import os # For environment variables
import boto3 # Add boto3 import
from botocore.exceptions import NoCredentialsError, ClientError
import time
import json
from datetime import datetime
from .utils.file_io import Status, parse_s3_uri, join_paths
import s3fs
from .utils.pipeline_registry import load_deployed_pipelines
from pyautocausal.orchestration.graph import ExecutableGraph

# --- pyautocausal Import ---
# This should work if pyautocausal is installed via poetry from the local path
# from pyautocausal.pipelines.example_graph import simple_graph
from pyautocausal.persistence.notebook_export import NotebookExporter

# Add standard Python logging
import logging

# Replace Celery logger with standard Python logger
logger = logging.getLogger(__name__)

# Configuration from Environment Variables
S3_OUTPUT_DIR = os.getenv("S3_BUCKET_OUTPUT")
S3_REGION = os.getenv("AWS_REGION", "us-east-1") # Default region if not set

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

# Near s3_client initialization, or where appropriate
s3 = s3fs.S3FileSystem(anon=False) # anon=False by default, uses boto3 credentials

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

# Worker will use a temporary local path for processing
WORKER_TEMP_DIR = Path("/tmp/pyautocausal_worker_space").resolve() # Or another suitable temp location

# Helper function to get file size
def _get_file_size(file_path: Path) -> int:
    """Helper function to get the size of a file."""
    try:
        return file_path.stat().st_size
    except FileNotFoundError:
        logger.warning(f"File not found when trying to get size: {file_path}")
        return 0 # Or raise an error, depending on desired behavior
    except Exception as e:
        logger.error(f"Error getting file size for {file_path}: {e}")
        return 0 # Or raise

def run_graph_job(job_id: str, input_s3_uri: str, graph: ExecutableGraph, column_mapping: dict, required_columns: list[str], task_store: dict) -> None:
    """
    FastAPI background task to download data from S3, run a given pipeline graph, and upload outputs to S3.
    This is a wrapper around the _run_graph_job function that updates the task store with the status and result.
    """
    # Update the task status in the task store
    task_store[job_id]["status"] = Status.RUNNING

    # Run the graph job
    try:
        output_s3_uri = _run_graph_job(job_id, input_s3_uri, graph, column_mapping, required_columns)
        task_store[job_id]["status"] = Status.SUCCESS
        task_store[job_id]["result"] = output_s3_uri
        return
    except Exception as e:
        # Log the error and update the task status
        logger.error(f"[{job_id}] Error in run_graph_job: {e}", exc_info=True)
        task_store[job_id]["status"] = Status.FAILED
        task_store[job_id]["error"] = str(e)
        raise  # Re-raise the error to be handled by FastAPI
    
def _run_graph_job(job_id: str, input_s3_uri: str, graph: ExecutableGraph, column_mapping: dict, required_columns: list[str]):
    """
    FastAPI background task to download data from S3, run a given pipeline graph, and upload outputs to S3.
    """
    task_start_time = time.time()
    logger.info(f"[{job_id}] FastAPI background task 'run_graph_job' started for input: {input_s3_uri}, graph: {graph.name if hasattr(graph, 'name') else 'unnamed'}")
    
    # Configuration validation
    if not S3_OUTPUT_DIR:
        logger.error(f"[{job_id}] S3_OUTPUT_BUCKET environment variable is not set. Cannot proceed.")
        raise ValueError("Server configuration error: S3 output bucket not specified.")
    
    # Parse the S3 output bucket URI to get bucket name and base path
    output_bucket_name, output_base_path = parse_s3_uri(S3_OUTPUT_DIR)
    if not output_bucket_name:
        logger.error(f"[{job_id}] Invalid S3 URI format for S3_OUTPUT_BUCKET: {S3_OUTPUT_DIR}")
        raise ValueError("Server configuration error: Invalid S3 output bucket URI format.")
    
    # Define local paths for this job on the worker instance
    local_output_job_path = Path(os.path.join(WORKER_TEMP_DIR, "output", job_id))
    # Ensure S3_OUTPUT_DIR has a trailing slash before appending job_id
    s3_output_key_prefix = f"{S3_OUTPUT_DIR.rstrip('/')}/{job_id}/"
    
    log_worker_event(
        event_type="job_processing_started",
        job_id=job_id, 
        input_s3_uri=input_s3_uri,
        output_s3_prefix=s3_output_key_prefix
    )

    try:
        # Create local directories for processing
        os.makedirs(local_output_job_path, exist_ok=True)

        # 1. Download data from S3 and load it
        logger.info(f"[{job_id}] Downloading and loading data from {input_s3_uri}.")
        if input_s3_uri.endswith(".csv"):
            data = pd.read_csv(input_s3_uri)
        else:
            raise ValueError(f"Input file must be a CSV file: {input_s3_uri}")

        # 2. Apply column mapping (rename)
        if column_mapping:
            data = data.rename(columns=column_mapping)

        # Validate required columns in renamed df
        required_cols = set(required_columns)
        if missing := required_cols - set(data.columns):
            raise ValueError(f"CSV missing required columns after mapping: {missing}")

        # 3. Configure runtime and fit graph
        logger.info(f"[{job_id}] Configuring runtime and fitting pipeline graph.")
        graph.configure_runtime(output_path=local_output_job_path)

        # Fit the loaded pipeline
        graph_start = time.time()
        graph.fit(df=data)
        
        graph_duration = (time.time() - graph_start) * 1000  # ms
        
        log_worker_event(
            event_type="graph_fitting_complete",
            job_id=job_id,
            duration_ms=graph_duration,
            nodes_count=len(graph.nodes) if hasattr(graph, 'nodes') else None
        )
        
        logger.info(f"[{job_id}] Graph fitting complete in {graph_duration:.2f}ms")

        # 4. Export Notebook locally
        logger.info(f"[{job_id}] Exporting graph to Jupyter Notebook locally.")
        try:
            notebook_start = time.time()
            exporter = NotebookExporter(graph)
            notebook_filename = "causal_analysis_notebook.ipynb"
            notebook_full_local_path = local_output_job_path / notebook_filename
            
            logger.debug(f"[{job_id}] Exporting notebook to {notebook_full_local_path}")
            exporter.export_notebook(notebook_full_local_path)
            
            notebook_duration = (time.time() - notebook_start) * 1000  # ms
            notebook_size = _get_file_size(notebook_full_local_path)
            
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

        logger.info(f"[{job_id}] Uploading results from {local_output_job_path} to S3 bucket {output_bucket_name} with prefix {s3_output_key_prefix}")

        upload_start = time.time()
        uploaded_files_count = 0
        total_uploaded_bytes = 0
        
        # Find all files to upload first
        output_files = list(local_output_job_path.rglob('*'))
        output_files = [f for f in output_files if f.is_file()]
        
        if not output_files:
            logger.warning(f"[{job_id}] No files found in {local_output_job_path} to upload to S3.")
        else:
            logger.info(f"[{job_id}] Found {len(output_files)} files in {local_output_job_path} to upload to S3")
        
        # Upload each file with progress tracking
        for item in output_files:
            if item.is_file():
                logger.info(f"Joining {s3_output_key_prefix.strip('/')} and {str(item.relative_to(local_output_job_path)).strip('/')} to get {s3_output_key_prefix.strip('/') + '/' + str(item.relative_to(local_output_job_path)).strip('/')}")
                s3_uri = s3_output_key_prefix.strip('/') + '/' + str(item.relative_to(local_output_job_path)).strip('/')
                file_size = item.stat().st_size
                    
                try:
                    logger.debug(f"[{job_id}] Uploading {item.name} ({file_size} bytes) to {s3_uri} using s3fs")
                    file_upload_start = time.time()
                    
                    s3.put(str(item), s3_uri) # Use s3fs.put()
                    
                    time_now = time.time()
                    file_upload_duration = (time_now - file_upload_start) * 1000  # ms
                    
                    log_worker_event(
                        event_type="file_upload_complete",
                        job_id=job_id,
                        duration_ms=file_upload_duration,
                        file_name=item.name,
                        file_size_bytes=file_size,
                        s3_key=s3_uri
                    )
                    
                    logger.debug(f"[{job_id}] Uploaded {item.name} to bucket: {output_bucket_name}, key: {s3_uri} in {file_upload_duration:.2f}ms")
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
            logger.info(f"[{job_id}] Successfully uploaded {uploaded_files_count} result files ({total_uploaded_bytes} bytes) to {s3_uri} in {upload_duration:.2f}ms")

        # Record total job time
        total_job_duration = (time.time() - task_start_time) * 1000  # ms
            
        log_worker_event(
            event_type="job_completed_successfully",
            job_id=job_id,
            duration_ms=total_job_duration,
            output_s3_uri=s3_output_key_prefix
        )
            
        logger.info(f"[{job_id}] Task 'run_graph_job' completed successfully in {total_job_duration:.2f}ms. Output at {s3_output_key_prefix}")
        return s3_output_key_prefix

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
        if os.path.exists(local_output_job_path):
            try:
                cleanup_start = time.time()
                shutil.rmtree(local_output_job_path)
                cleanup_duration = (time.time() - cleanup_start) * 1000  # ms
                
                log_worker_event(
                    event_type="cleanup_complete",
                    job_id=job_id,
                    duration_ms=cleanup_duration,
                    directory=str(local_output_job_path)
                )
                logger.info(f"[{job_id}] Cleaned up local temporary directory: {local_output_job_path} in {cleanup_duration:.2f}ms")
            except Exception as e_clean:
                logger.error(f"[{job_id}] Error cleaning up local temporary directory {local_output_job_path}: {e_clean}", exc_info=True)