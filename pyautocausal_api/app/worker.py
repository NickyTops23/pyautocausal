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

_s3_fs_instance = None

def get_s3_fs():
    """Get a singleton s3fs.S3FileSystem instance."""
    global _s3_fs_instance
    if _s3_fs_instance is None:
        # Use anon=False to ensure boto3 credentials are used
        _s3_fs_instance = s3fs.S3FileSystem(anon=False)
    return _s3_fs_instance

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

        s3 = get_s3_fs()

        # 1. Download data from S3 and load it
        logger.info(f"[{job_id}] Downloading and loading data from {input_s3_uri}.")
        if input_s3_uri.endswith(".csv"):
            with s3.open(input_s3_uri, 'r') as f:
                data = pd.read_csv(f)
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

        # --- New: Zip results and upload as a single archive ---
        logger.info(f"[{job_id}] Zipping results from {local_output_job_path} for upload.")
        
        try:
            # Create a zip archive of the output directory
            zip_file_name = f"pyautocausal_results_{job_id}"
            # Place archive in parent of output dir to avoid zipping the archive itself
            archive_path_base = os.path.join(WORKER_TEMP_DIR, "output", zip_file_name)
            archive_local_path_str = shutil.make_archive(
                base_name=archive_path_base,
                format='zip',
                root_dir=local_output_job_path
            )
            archive_local_path = Path(archive_local_path_str)
            archive_file_size = _get_file_size(archive_local_path)
            
            # Define S3 path for the zip file
            output_bucket_name, output_base_path = parse_s3_uri(S3_OUTPUT_DIR)
            archive_s3_key = f"{output_base_path.strip('/')}/{job_id}/{archive_local_path.name}"
            archive_s3_uri = f"s3://{output_bucket_name}/{archive_s3_key}"

            logger.info(f"[{job_id}] Uploading result archive ({archive_file_size} bytes) to {archive_s3_uri}")
            upload_start = time.time()
            
            s3.put(str(archive_local_path), archive_s3_uri)
            
            upload_duration = (time.time() - upload_start) * 1000
            
            log_worker_event(
                event_type="archive_upload_complete",
                job_id=job_id,
                duration_ms=upload_duration,
                archive_size_bytes=archive_file_size,
                s3_uri=archive_s3_uri
            )
            logger.info(f"[{job_id}] Successfully uploaded result archive to {archive_s3_uri} in {upload_duration:.2f}ms")

        except Exception as e:
            logger.error(f"[{job_id}] Failed to create or upload result archive: {e}", exc_info=True)
            raise  # Re-raise to fail the job
        
        # Record total job time
        total_job_duration = (time.time() - task_start_time) * 1000  # ms
            
        log_worker_event(
            event_type="job_completed_successfully",
            job_id=job_id,
            duration_ms=total_job_duration,
            output_s3_uri=archive_s3_uri
        )
            
        logger.info(f"[{job_id}] Task 'run_graph_job' completed successfully in {total_job_duration:.2f}ms. Output at {archive_s3_uri}")
        return archive_s3_uri

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