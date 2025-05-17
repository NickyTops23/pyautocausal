from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Path as FastAPIPath, Request, BackgroundTasks
from celery.result import AsyncResult
from pydantic import BaseModel, Field
import uuid
from pathlib import Path
import os # For environment variables
import boto3 # Add boto3 import
from botocore.exceptions import NoCredentialsError, ClientError
import time
import logging
import json
from datetime import datetime
import io
from enum import Enum
# Import the Celery app instance and the task definition from app.worker
# This assumes main.py and worker.py are in the same 'app' directory.
from .worker import run_graph_job

# Import our file utilities
from .utils.file_io import (
    is_s3_path, parse_path, extract_filename, 
    check_path_exists, read_from, write_to,
    join_paths,
    Status
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("fastapi_app")
logger.setLevel(logging.DEBUG)  # Set to DEBUG for maximum details during troubleshooting

# Optional: Add a file handler for persistent logs
# file_handler = logging.FileHandler("/var/log/pyautocausal_api.log")
# file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
# logger.addHandler(file_handler)

app = FastAPI(
    title="PyAutoCausal Jobs API",
    description="API for submitting and monitoring PyAutoCausal graph processing jobs.",
    version="0.1.0"
)

job_status_store = {}

# Configuration from Environment Variables
S3_INPUT_DIR = os.getenv("S3_BUCKET_INPUT")
S3_OUTPUT_DIR = os.getenv("S3_BUCKET_OUTPUT")
S3_REGION = os.getenv("AWS_REGION", "us-east-1") # Default region if not set

# Log environment configuration at startup
logger.info(f"Starting PyAutoCausal API with: S3_BUCKET_INPUT={S3_INPUT_DIR}, S3_BUCKET_OUTPUT={S3_BUCKET_OUTPUT}, AWS_REGION={S3_REGION}")

# Configure boto3 with timeouts
s3_client = boto3.client(
    "s3", 
    region_name=S3_REGION,
    config=boto3.session.Config(
        connect_timeout=5,  # 5 seconds
        read_timeout=60,    # 60 seconds
        retries={'max_attempts': 3}
    )
)

# Function to parse S3 URI into bucket and path components
def parse_s3_uri(s3_uri):
    """Parse an S3 URI into bucket and key components."""
    logger.debug(f"Parsing S3 URI: {s3_uri}")
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

# Remove local BASE_INPUT_PATH, as we're using S3 now
# BASE_INPUT_PATH = Path("./local_job_files/inputs").resolve()
# BASE_INPUT_PATH.mkdir(parents=True, exist_ok=True)

# --- Pydantic Models for API Request/Response ---
class JobSubmissionResponse(BaseModel):
    job_id: str
    status_url: str # Fully qualified URL for status checking
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str # Celery states: PENDING, STARTED, RETRY, FAILURE, SUCCESS, or custom PROCESSING
    message: str | None = None
    result_path: str | None = None # Path to results if job COMPLETED
    error_details: str | None = None

class InputPathSubmission(BaseModel):
    input_path: str = Field(..., description="Path to input file (S3 URI or local path)")
    original_filename: str | None = Field(None, description="Original filename (optional, extracted from path if not provided)")

# Helper function for structured logging
def log_event(event_type, job_id=None, duration_ms=None, status=None, error=None, **kwargs):
    """Log a structured event with consistent format"""
    log_data = {
        "event": event_type,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    if job_id:
        log_data["job_id"] = job_id
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


# --- Unified API Endpoint ---
@app.post("/jobs", response_model=JobSubmissionResponse, status_code=202)
async def submit_job(
    request: Request, # To construct full status URL
    input_path: str,
    background_tasks: BackgroundTasks
):
    """
    Submit a new job to process a data file with the PyAutoCausal simple_graph.
    
    You can either:
    1. Upload a file directly using the 'file' parameter, or
    2. Specify the path to an existing file using 'input_path' (can be S3 URI or local path)
    """
    start_time = time.time()
    job_id = str(uuid.uuid4())
    
    # Validate that we have either a file upload or an input path
    if not input_path:
        logger.warning(f"[{job_id}] Job submission missing both file and input_path")
        raise HTTPException(status_code=400, detail="You must provide either a file upload or an input_path")
    
    # Read the file from the input path
    file_content = read_from(input_path)

    # Determine original filename and set up paths
    original_filename = extract_filename(input_path)
    logger.info(f"[{job_id}] New job submission started for input path: {input_path}")
    
    # Configure the input path where the worker will find the file
    if not S3_INPUT_DIR:
        logger.error(f"[{job_id}] S3_INPUT_BUCKET environment variable is not set.")
        raise HTTPException(status_code=500, detail="Server configuration error: S3 input bucket not specified.")
        
    # Generate destination S3 path for uploaded file
    dest_path = join_paths(S3_INPUT_DIR, f"inputs/{job_id}/{original_filename}")
        
    try:
        # Upload the file
        upload_start = time.time()
        logger.debug(f"[{job_id}] Starting upload to {dest_path}")
        
        # Write the uploaded file to destination
        write_to(file_content, dest_path)
        
        # Set input_path to the location we just wrote to
        input_path_for_worker = dest_path
        
        upload_duration = (time.time() - upload_start) * 1000  # ms
        log_event(
            "file_upload_complete", 
            job_id=job_id, 
            duration_ms=upload_duration,
            file_name=original_filename,
            destination_path=dest_path
        )
    except Exception as e:
        logger.error(f"[{job_id}] Error during file upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during file upload: {str(e)}")
    
    try:
        path_check_start = time.time()
        if not check_path_exists(input_path_for_worker):
            logger.warning(f"[{job_id}] File does not exist: {input_path_for_worker}")
            raise HTTPException(status_code=404, detail=f"The specified file does not exist: {input_path}")
        
        path_check_duration = (time.time() - path_check_start) * 1000  # ms
        logger.debug(f"[{job_id}] Input file exists in S3, check took {path_check_duration:.2f}ms")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise  # Re-raise HTTP exceptions
        logger.error(f"[{job_id}] Error checking intermediate storage path: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error checking intermediate storage path: {str(e)}")

    # Dispatch the FastAPI Background task with the input path
    try:
        job_start = time.time()
        logger.debug(f"[{job_id}] Dispatching FastAPI background task with input: {input_path}")
        # Use FastAPI's BackgroundTasks instead of Celery
        background_tasks.add_task(
            run_graph_job,
            job_id=job_id, 
            input_s3_uri=input_path_for_worker, 
            task_store=job_status_store
        )
        
        task_init_duration = (time.time() - job_start) * 1000  # ms
        
        log_event(
            "task_submitted", 
            job_id=job_id, 
            duration_ms=task_init_duration,
            input_path=input_path
        )

    except Exception as e:
        logger.error(f"[{job_id}] Failed to submit job to FastAPI background task: {e}", exc_info=True)
        log_event(
            "task_submission_failed", 
            job_id=job_id, 
            error=str(e)
        )
        raise HTTPException(status_code=503, detail=f"Job queueing service unavailable: {e}")

    job_status_store[job_id] = {
        "status": Status.PENDING,
        "original_filename": original_filename,
        "input_path": input_path,
        "result": None,
        "error": None
    }

    # Construct the full status URL using the request object
    status_url = request.url_for("get_job_status", job_id=job_id)
    
    total_duration = (time.time() - start_time) * 1000  # ms
    log_event(
        "job_submission_complete", 
        job_id=job_id, 
        duration_ms=total_duration,
        status_url=str(status_url)
    )

    return JobSubmissionResponse(
        job_id=job_id,
        status_url=str(status_url),
        message="Job submitted successfully. Check the status URL for updates."
    )

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str = FastAPIPath(..., description="The ID of the job to check.")):
    """
    Retrieve the status and results (if available) of a previously submitted job.
    """
    start_time = time.time()
    logger.debug(f"[{job_id}] Status check requested")
    
    job_info = job_status_store.get(job_id)

    if not job_info:
        logger.warning(f"[{job_id}] Job ID not found in job_status_store")
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found. It may not have been submitted or has been cleared.")

    try:
        current_celery_status = task_result.status
        logger.debug(f"[{job_id}] Current Celery status: {current_celery_status}")
    except Exception as e:
        logger.error(f"[{job_id}] Error retrieving task status from Celery: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving task status: {str(e)}")

    user_friendly_status = current_celery_status # Default
    result_data_path = None
    user_message = f"Job '{job_id}' is currently {current_celery_status}."
    error_info_details = None

    if task_result.successful(): # Equivalent to current_celery_status == "SUCCESS"
        user_friendly_status = "COMPLETED"
        result_data_path = str(task_result.result) # This is the path string returned by the worker
        user_message = f"Job '{job_id}' completed successfully."
        logger.info(f"[{job_id}] Job completed successfully, result path: {result_data_path}")
    elif task_result.failed(): # Equivalent to current_celery_status == "FAILURE"
        user_friendly_status = "FAILED"
        error_exception = task_result.result # This is the exception object
        user_message = f"Job '{job_id}' failed."
        error_info_details = f"{type(error_exception).__name__}: {str(error_exception)}"
        logger.error(f"[{job_id}] Job failed: {error_info_details}")
    elif current_celery_status == "PENDING":
        logger.info(f"[{job_id}] Job is pending/queued")
        user_message = f"Job '{job_id}' is queued and waiting for a worker."
    elif current_celery_status == "STARTED":
        logger.info(f"[{job_id}] Job has started on a worker")
        user_message = f"Job '{job_id}' has been picked up by a worker and is running."
    elif current_celery_status == "RETRY":
        logger.info(f"[{job_id}] Job is being retried")
        user_message = f"Job '{job_id}' is being retried by a worker."
    # Check for custom states from worker's self.update_state()
    elif isinstance(task_result.info, dict) and task_result.state == 'PROCESSING': # Our custom state
        user_friendly_status = "PROCESSING"
        status_message = task_result.info.get('message', f"Job is processing...")
        user_message = status_message
        logger.info(f"[{job_id}] Job is processing: {status_message}")
    elif task_result.state not in ["PENDING", "STARTED", "SUCCESS", "FAILURE", "RETRY"]: # Other Celery states
        user_message = f"Job '{job_id}' is in an intermediate state: {task_result.state}."
        if isinstance(task_result.info, dict):
            user_message += f" Details: {task_result.info.get('message', 'No specific details.')}"
        logger.info(f"[{job_id}] Job in intermediate state: {task_result.state}")

    total_duration = (time.time() - start_time) * 1000  # ms
    log_event(
        "job_status_check", 
        job_id=job_id,
        duration_ms=total_duration,
        celery_status=current_celery_status,
        user_status=user_friendly_status
    )

    return JobStatusResponse(
        job_id=job_id,
        status=user_friendly_status,
        message=user_message,
        result_path=result_data_path,
        error_details=error_info_details
    )

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API and Celery connectivity
    """
    start_time = time.time()
    health_status = {
        "api": "healthy",
        "s3": "unknown",
    }
    
    # Check S3 connectivity if we have S3 configuration
    if S3_INPUT_DIR:
        try:
            s3_start = time.time()
            # Check if bucket exists by parsing and checking
            scheme, bucket, _ = parse_path(S3_INPUT_DIR)
            if scheme == "s3":
                # Try a simple operation to test connectivity
                from .utils.file_io import s3_client
                s3_client.list_buckets()
                s3_duration = (time.time() - s3_start) * 1000  # ms
                health_status["s3"] = "healthy"
                health_status["s3_response_time_ms"] = s3_duration
        except Exception as e:
            health_status["s3"] = "unhealthy"
            health_status["s3_error"] = str(e)
    
    # Overall health determination
    health_status["overall"] = "healthy" if all(
        status == "healthy" for _, status in health_status.items() 
    ) else "unhealthy"
    
    health_status["timestamp"] = datetime.utcnow().isoformat()
    health_status["response_time_ms"] = (time.time() - start_time) * 1000  # ms
    
    # Log the health check
    log_level = logging.INFO if health_status["overall"] == "healthy" else logging.WARNING
    logger.log(log_level, f"Health check: {health_status['overall']}", extra=health_status)
    
    return health_status

# Root endpoint for API health check
@app.get("/")
async def read_root():
    logger.debug("Root endpoint accessed")
    return {"message": "PyAutoCausal API is running. Submit jobs to /jobs."}
