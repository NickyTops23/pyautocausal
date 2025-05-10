from fastapi import FastAPI, UploadFile, File, HTTPException, Path as FastAPIPath, Request
from celery.result import AsyncResult
from pydantic import BaseModel
import uuid
from pathlib import Path
import shutil
import os # For environment variables
import boto3 # Add boto3 import
from botocore.exceptions import NoCredentialsError, ClientError

# Import the Celery app instance and the task definition from app.worker
# This assumes main.py and worker.py are in the same 'app' directory.
from .worker import celery_app, run_graph_job

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
    if not S3_INPUT_BUCKET:
        logger.error("S3_INPUT_BUCKET environment variable is not set.")
        raise HTTPException(status_code=500, detail="Server configuration error: S3 input bucket not specified.")

    # Parse the S3 input bucket URI to get bucket name and base path
    bucket_name, base_path = parse_s3_uri(S3_INPUT_BUCKET)
    if not bucket_name:
        logger.error(f"Invalid S3 URI format for S3_INPUT_BUCKET: {S3_INPUT_BUCKET}")
        raise HTTPException(status_code=500, detail="Server configuration error: Invalid S3 input bucket URI format.")

    job_id = str(uuid.uuid4())
    original_filename = file.filename or "uploaded_file"
    
    # Construct the full S3 key including the base path if it exists
    if base_path:
        # Ensure base_path has trailing slash but not leading slash
        base_path = base_path.strip('/')
        if base_path:
            base_path += '/'
        s3_input_key = f"{base_path}inputs/{job_id}/{original_filename}"
    else:
        s3_input_key = f"inputs/{job_id}/{original_filename}"

    try:
        # Upload the file to S3
        s3_client.upload_fileobj(file.file, bucket_name, s3_input_key)
        input_s3_uri = f"s3://{bucket_name}/{s3_input_key}"
        logger.info(f"File for job {job_id} uploaded to {input_s3_uri}")

    except NoCredentialsError:
        logger.error("AWS credentials not found for S3 upload.")
        raise HTTPException(status_code=500, detail="Server configuration error: AWS credentials not found.")
    except ClientError as e:
        logger.error(f"S3 upload failed for job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not upload file to S3: {e}")
    finally:
        await file.close()

    # Dispatch the Celery task with the S3 URI
    try:
        task = run_graph_job.delay(job_id=job_id, input_s3_uri=input_s3_uri, original_filename=original_filename)
    except Exception as e:
        logger.error(f"Failed to submit job {job_id} to Celery queue: {e}", exc_info=True)
        # Attempt to delete the uploaded S3 object if task submission fails
        try:
            s3_client.delete_object(Bucket=bucket_name, Key=s3_input_key)
            logger.info(f"Cleaned up S3 object {input_s3_uri} due to Celery submission failure.")
        except Exception as s3_del_e:
            logger.error(f"Failed to clean up S3 object {input_s3_uri}: {s3_del_e}")
        raise HTTPException(status_code=503, detail=f"Job queueing service unavailable: {e}")

    job_to_celery_map[job_id] = {
        "celery_task_id": task.id,
        "original_filename": original_filename,
        "input_s3_uri": input_s3_uri
    }

    # Construct the full status URL using the request object
    status_url = request.url_for("get_job_status", job_id=job_id)

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
    job_info = job_to_celery_map.get(job_id)

    if not job_info:
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found. It may not have been submitted or has been cleared from the PoC map.")

    celery_task_id = job_info.get("celery_task_id")
    if not celery_task_id:
        # This internal state error should ideally not happen
        logger.error(f"Internal error: Celery task ID not found for job_id '{job_id}' in job_to_celery_map.")
        raise HTTPException(status_code=500, detail=f"Internal error retrieving task details for job '{job_id}'.")

    task_result = AsyncResult(celery_task_id, app=celery_app)

    current_celery_status = task_result.status
    user_friendly_status = current_celery_status # Default
    result_data_path = None
    user_message = f"Job '{job_id}' is currently {current_celery_status}."
    error_info_details = None

    if task_result.successful(): # Equivalent to current_celery_status == "SUCCESS"
        user_friendly_status = "COMPLETED"
        result_data_path = str(task_result.result) # This is the path string returned by the worker
        user_message = f"Job '{job_id}' completed successfully."
    elif task_result.failed(): # Equivalent to current_celery_status == "FAILURE"
        user_friendly_status = "FAILED"
        error_exception = task_result.result # This is the exception object
        user_message = f"Job '{job_id}' failed."
        error_info_details = f"{type(error_exception).__name__}: {str(error_exception)}"
        # For very detailed server-side logging of the traceback:
        # logger.error(f"Traceback for failed job {job_id} (task {celery_task_id}): {task_result.traceback}")
    elif current_celery_status == "PENDING":
        user_message = f"Job '{job_id}' is queued and waiting for a worker."
    elif current_celery_status == "STARTED":
        user_message = f"Job '{job_id}' has been picked up by a worker and is running."
    elif current_celery_status == "RETRY":
        user_message = f"Job '{job_id}' is being retried by a worker."
    # Check for custom states from worker's self.update_state()
    elif isinstance(task_result.info, dict) and task_result.state == 'PROCESSING': # Our custom state
        user_friendly_status = "PROCESSING"
        user_message = task_result.info.get('message', f"Job '{job_id}' is processing...")
    elif task_result.state not in ["PENDING", "STARTED", "SUCCESS", "FAILURE", "RETRY"]: # Other Celery states
        user_message = f"Job '{job_id}' is in an intermediate state: {task_result.state}."
        if isinstance(task_result.info, dict):
            user_message += f" Details: {task_result.info.get('message', 'No specific details.')}"


    return JobStatusResponse(
        job_id=job_id,
        status=user_friendly_status,
        message=user_message,
        result_path=result_data_path,
        error_details=error_info_details
    )

# It's good practice to have a root endpoint for basic API health check
@app.get("/")
async def read_root():
    return {"message": "PyAutoCausal API is running. Submit jobs to /jobs."}

# --- Optional: Add logging configuration for FastAPI if needed ---
# (Celery has its own logger, configured via get_task_logger)
import logging
logger = logging.getLogger("fastapi_app") # General logger for the API app itself
# Configure as needed, e.g., logging.basicConfig(level=logging.INFO) at startup
# or use Uvicorn's logging.
