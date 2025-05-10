from fastapi import FastAPI, UploadFile, File, HTTPException, Path as FastAPIPath, Request
from celery.result import AsyncResult
from pydantic import BaseModel
import uuid
from pathlib import Path
import shutil

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
# Value: {"celery_task_id": str, "original_filename": str}
# For a PoC, this is okay. For production, use a persistent store (e.g., Redis, DB).
job_to_celery_map = {}

# Define base paths for our local "blob storage" (inputs)
# These paths are relative to where the main.py (FastAPI app) is run.
# If you run uvicorn from `pyautocausal_api/`, these will be correct.
BASE_INPUT_PATH = Path("./local_job_files/inputs").resolve()

# Ensure base input directory exists at startup (worker handles output dir)
BASE_INPUT_PATH.mkdir(parents=True, exist_ok=True)

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
    """
    job_id = str(uuid.uuid4())

    # Create a subdirectory for this job's input file
    job_input_dir = BASE_INPUT_PATH / job_id
    job_input_dir.mkdir(parents=True, exist_ok=True)

    input_file_path = job_input_dir / (file.filename or "uploaded_file") # Handle case with no filename

    try:
        # Save the uploaded file to the job-specific input directory
        with open(input_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        # Clean up job input directory if file save fails
        if job_input_dir.exists():
            shutil.rmtree(job_input_dir, ignore_errors=True)
        logger.error(f"Error saving uploaded file for job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not save uploaded file '{file.filename}': {e}")
    finally:
        await file.close() # Important to close the file stream

    # Dispatch the Celery task
    try:
        # Pass our job_id and the original filename.
        # The worker constructs the full path using BASE_INPUT_PATH.
        task = run_graph_job.delay(job_id=job_id, original_filename=file.filename or "uploaded_file")
    except Exception as e: # Catch potential issues with sending task to broker (e.g., Redis down)
        if job_input_dir.exists(): # Clean up if task submission fails
            shutil.rmtree(job_input_dir, ignore_errors=True)
        logger.error(f"Failed to submit job {job_id} to Celery queue: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Job queueing service unavailable: {e}")

    # Store the mapping from our job_id to Celery's task.id
    job_to_celery_map[job_id] = {"celery_task_id": task.id, "original_filename": file.filename or "uploaded_file"}

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
