from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Path as FastAPIPath, Request
from celery.result import AsyncResult
from pydantic import BaseModel, Field
import uuid
from pathlib import Path
import shutil
import os # For environment variables
import boto3 # Add boto3 import
from botocore.exceptions import NoCredentialsError, ClientError
import time
import logging
import json
from datetime import datetime

# Import the Celery app instance and the task definition from app.worker
# This assumes main.py and worker.py are in the same 'app' directory.
from .worker import celery_app, run_graph_job

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

# In-memory store for mapping our job_id to Celery's task_id
# Key: our application-specific job_id (str)
# Value: {"celery_task_id": str, "original_filename": str, "input_s3_uri": str}
# For a PoC, this is okay. For production, use a persistent store (e.g., Redis, DB).
job_to_celery_map = {}

# Configuration from Environment Variables
S3_INPUT_BUCKET = os.getenv("S3_INPUT_BUCKET")
S3_REGION = os.getenv("AWS_REGION", "us-east-1") # Default region if not set

# Log environment configuration at startup
logger.info(f"Starting PyAutoCausal API with: S3_INPUT_BUCKET={S3_INPUT_BUCKET}, AWS_REGION={S3_REGION}")

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
class S3JobSubmission(BaseModel):
    s3_uri: str = Field(..., description="S3 URI of the file to process (e.g., s3://bucket/path/to/file.csv)")
    original_filename: str | None = Field(None, description="Original filename (optional, extracted from S3 path if not provided)")

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

# --- API Endpoints ---
@app.post("/jobs", response_model=JobSubmissionResponse, status_code=202)
async def submit_job(
    request: Request, # To construct full status URL
    file: UploadFile = File(..., description="The data file (e.g., CSV) to process.")
):
    """
    Submit a new job to process a data file with the PyAutoCausal simple_graph.
    The file will be uploaded to S3, and its S3 URI will be passed to the worker.
    """
    start_time = time.time()
    job_id = str(uuid.uuid4())
    original_filename = file.filename or "uploaded_file"
    
    logger.info(f"[{job_id}] New job submission started for file: {original_filename}")
    
    if not S3_INPUT_BUCKET:
        logger.error(f"[{job_id}] S3_INPUT_BUCKET environment variable is not set.")
        raise HTTPException(status_code=500, detail="Server configuration error: S3 input bucket not specified.")

    # Parse the S3 input bucket URI to get bucket name and base path
    bucket_name, base_path = parse_s3_uri(S3_INPUT_BUCKET)
    if not bucket_name:
        logger.error(f"[{job_id}] Invalid S3 URI format for S3_INPUT_BUCKET: {S3_INPUT_BUCKET}")
        raise HTTPException(status_code=500, detail="Server configuration error: Invalid S3 input bucket URI format.")

    # Construct the full S3 key including the base path if it exists
    if base_path:
        # Ensure base_path has trailing slash but not leading slash
        base_path = base_path.strip('/')
        if base_path:
            base_path += '/'
        s3_input_key = f"{base_path}inputs/{job_id}/{original_filename}"
    else:
        s3_input_key = f"inputs/{job_id}/{original_filename}"
    
    logger.debug(f"[{job_id}] Constructed S3 key: {s3_input_key}")

    try:
        # Upload the file to S3
        upload_start = time.time()
        logger.debug(f"[{job_id}] Starting S3 upload to bucket {bucket_name}")
        s3_client.upload_fileobj(file.file, bucket_name, s3_input_key)
        upload_duration = (time.time() - upload_start) * 1000  # ms
        input_s3_uri = f"s3://{bucket_name}/{s3_input_key}"
        
        log_event(
            "s3_upload_complete", 
            job_id=job_id, 
            duration_ms=upload_duration,
            file_name=original_filename,
            s3_uri=input_s3_uri
        )

    except NoCredentialsError as e:
        logger.error(f"[{job_id}] AWS credentials not found for S3 upload: {e}")
        raise HTTPException(status_code=500, detail="Server configuration error: AWS credentials not found.")
    except ClientError as e:
        logger.error(f"[{job_id}] S3 upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not upload file to S3: {e}")
    except Exception as e:
        logger.error(f"[{job_id}] Unexpected error during S3 upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error during file upload: {str(e)}")
    finally:
        await file.close()

    # Dispatch the Celery task with the S3 URI
    try:
        celery_start = time.time()
        logger.debug(f"[{job_id}] Dispatching Celery task with input: {input_s3_uri}")
        task = run_graph_job.delay(job_id=job_id, input_s3_uri=input_s3_uri, original_filename=original_filename)
        celery_duration = (time.time() - celery_start) * 1000  # ms
        
        log_event(
            "celery_task_submitted", 
            job_id=job_id, 
            duration_ms=celery_duration,
            celery_task_id=task.id
        )
        
    except Exception as e:
        logger.error(f"[{job_id}] Failed to submit job to Celery queue: {e}", exc_info=True)
        # Attempt to delete the uploaded S3 object if task submission fails
        try:
            logger.debug(f"[{job_id}] Cleaning up S3 object due to Celery submission failure")
            s3_client.delete_object(Bucket=bucket_name, Key=s3_input_key)
            logger.info(f"[{job_id}] Cleaned up S3 object {input_s3_uri} due to Celery submission failure")
        except Exception as s3_del_e:
            logger.error(f"[{job_id}] Failed to clean up S3 object {input_s3_uri}: {s3_del_e}")
        
        log_event(
            "celery_task_submission_failed", 
            job_id=job_id, 
            error=str(e)
        )
        raise HTTPException(status_code=503, detail=f"Job queueing service unavailable: {e}")

    job_to_celery_map[job_id] = {
        "celery_task_id": task.id,
        "original_filename": original_filename,
        "input_s3_uri": input_s3_uri
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
    
    job_info = job_to_celery_map.get(job_id)

    if not job_info:
        logger.warning(f"[{job_id}] Job ID not found in job_to_celery_map")
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found. It may not have been submitted or has been cleared from the PoC map.")

    celery_task_id = job_info.get("celery_task_id")
    if not celery_task_id:
        # This internal state error should ideally not happen
        logger.error(f"[{job_id}] Internal error: Celery task ID not found in job_to_celery_map entry")
        raise HTTPException(status_code=500, detail=f"Internal error retrieving task details for job '{job_id}'.")

    try:
        logger.debug(f"[{job_id}] Retrieving task result for Celery task ID: {celery_task_id}")
        task_result = AsyncResult(celery_task_id, app=celery_app)
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
        # For very detailed server-side logging of the traceback:
        logger.debug(f"[{job_id}] Traceback for failed job (task {celery_task_id}): {task_result.traceback}")
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
    return {"status": "healthy"}

# Add this new endpoint
@app.post("/jobs/s3", response_model=JobSubmissionResponse, status_code=202)
async def submit_job_from_s3(
    request: Request,
    submission: S3JobSubmission
):
    """
    Submit a new job to process a data file that already exists in S3.
    Provide the full S3 URI to the file.
    """
    start_time = time.time()
    s3_uri = submission.s3_uri
    job_id = str(uuid.uuid4())
    
    logger.info(f"[{job_id}] New S3 job submission started for URI: {s3_uri}")
    
    # Validate the S3 URI format
    source_bucket, source_key = parse_s3_uri(s3_uri)
    if not source_bucket or not source_key:
        logger.warning(f"[{job_id}] Invalid S3 URI format: {s3_uri}")
        raise HTTPException(status_code=400, detail="Invalid S3 URI format. Must be s3://bucket/path/to/file")
    
    # Extract filename from S3 path if not provided
    original_filename = submission.original_filename
    if not original_filename:
        original_filename = Path(source_key).name
        logger.debug(f"[{job_id}] Using filename from S3 path: {original_filename}")
    
    # Check if the S3 object exists
    try:
        logger.debug(f"[{job_id}] Checking if S3 object exists: {s3_uri}")
        check_start = time.time()
        s3_client.head_object(Bucket=source_bucket, Key=source_key)
        check_duration = (time.time() - check_start) * 1000  # ms
        logger.debug(f"[{job_id}] S3 object exists, check took {check_duration:.2f}ms")
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "404":
            logger.warning(f"[{job_id}] S3 object does not exist: {s3_uri}")
            raise HTTPException(status_code=404, detail=f"The specified S3 object does not exist: {s3_uri}")
        else:
            logger.error(f"[{job_id}] Error accessing S3 object {s3_uri}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error accessing S3 object: {e}")
    
    # Dispatch the Celery task directly with the provided S3 URI
    try:
        celery_start = time.time()
        logger.debug(f"[{job_id}] Dispatching Celery task with input: {s3_uri}")
        task = run_graph_job.delay(job_id=job_id, input_s3_uri=s3_uri, original_filename=original_filename)
        celery_duration = (time.time() - celery_start) * 1000  # ms
        
        log_event(
            "celery_task_submitted", 
            job_id=job_id, 
            duration_ms=celery_duration,
            celery_task_id=task.id,
            s3_uri=s3_uri
        )
    except Exception as e:
        logger.error(f"[{job_id}] Failed to submit job to Celery queue: {e}", exc_info=True)
        log_event(
            "celery_task_submission_failed", 
            job_id=job_id, 
            error=str(e)
        )
        raise HTTPException(status_code=503, detail=f"Job queueing service unavailable: {e}")

    job_to_celery_map[job_id] = {
        "celery_task_id": task.id,
        "original_filename": original_filename,
        "input_s3_uri": s3_uri
    }

    # Construct the full status URL
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

# Add a health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API and Celery connectivity
    """
    start_time = time.time()
    health_status = {
        "api": "healthy",
        "celery": "unknown",
        "s3": "unknown",
    }
    
    # Check Celery connectivity
    try:
        celery_start = time.time()
        # Simple ping by inspecting stats - doesn't require actual task submission
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        celery_duration = (time.time() - celery_start) * 1000  # ms
        
        if stats:
            health_status["celery"] = "healthy"
            health_status["celery_workers"] = len(stats)
            health_status["celery_response_time_ms"] = celery_duration
        else:
            health_status["celery"] = "unhealthy"
            health_status["celery_error"] = "No workers available"
    except Exception as e:
        health_status["celery"] = "unhealthy"
        health_status["celery_error"] = str(e)
    
    # Check S3 connectivity
    try:
        s3_start = time.time()
        s3_client.list_buckets()
        s3_duration = (time.time() - s3_start) * 1000  # ms
        health_status["s3"] = "healthy"
        health_status["s3_response_time_ms"] = s3_duration
    except Exception as e:
        health_status["s3"] = "unhealthy"
        health_status["s3_error"] = str(e)
    
    # Overall health determination
    health_status["overall"] = "healthy" if all(
        status == "healthy" for key, status in health_status.items() 
        if key in ["api", "celery", "s3"]
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
