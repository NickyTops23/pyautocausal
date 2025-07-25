from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Path as FastAPIPath, Request, BackgroundTasks
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
from .utils.file_io import s3_client
from fastapi.middleware.cors import CORSMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
import yaml
from .utils.pipeline_registry import load_deployed_pipelines
from pyautocausal.orchestration.graph import ExecutableGraph
import s3fs
import tempfile

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

# Add CORS middleware to allow requests from the front-end running on localhost:9000 (and other origins during development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:9000",  # local dev server for the UI
        "http://127.0.0.1:9000",  # alternative localhost notation
        "http://localhost:5173",  # Vite dev server
        "*"  # TODO: tighten this list in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    ProxyHeadersMiddleware,
    trusted_hosts="*"
)

job_status_store = {}

# Configuration from Environment Variables
S3_OUTPUT_DIR = os.getenv("S3_BUCKET_OUTPUT")
S3_REGION = os.getenv("AWS_REGION", "us-east-1") # Default region if not set

# Log environment configuration at startup
logger.info(f"Starting PyAutoCausal API with: S3_BUCKET_OUTPUT={S3_OUTPUT_DIR}, AWS_REGION={S3_REGION}")

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
s3 = s3fs.S3FileSystem(anon=False) # For loading graph objects

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
    result_s3_uri: str | None = None
    download_url: str | None = None # Pre-signed URL for downloading results
    error_details: str | None = None

class JobSubmissionRequest(BaseModel):
    input_path: str
    pipeline_name: str
    column_mapping: dict

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
    request: Request,
    background_tasks: BackgroundTasks, # To construct full status URL
    job_details: JobSubmissionRequest,
):
    """
    Submit a new job to process a data file with the PyAutoCausal simple_graph.
    
    The request body should be a JSON object with the following fields:
    - input_path: Path to input file (S3 URI)
    - pipeline_name: Name of the pipeline to use
    - column_mapping: JSON object mapping user CSV columns to pipeline columns
    """
    start_time = time.time()
    job_id = str(uuid.uuid4())

    input_path = job_details.input_path
    pipeline_name = job_details.pipeline_name
    mapping_dict = job_details.column_mapping
    
    # Basic validation of required form fields
    if not input_path:
        raise HTTPException(status_code=400, detail="input_path is required")

    # Validate pipeline_name exists
    pipelines_dict = load_deployed_pipelines()
    if pipeline_name not in pipelines_dict:
        raise HTTPException(status_code=400, detail=f"Unknown pipeline_name '{pipeline_name}'.")

    pipeline_meta = pipelines_dict[pipeline_name]
    graph_uri = pipeline_meta.get("graph_uri")
    if not graph_uri:
        raise HTTPException(status_code=500, detail=f"Graph URI not configured for pipeline '{pipeline_name}'.")

    # Ensure all required columns are mapped
    required_cols = set(pipeline_meta["required_columns"])
    mapped_values = set(mapping_dict.values())
    missing_required = required_cols - mapped_values
    if missing_required:
        raise HTTPException(status_code=422, detail={
            "error": "Missing required column mappings",
            "missing_columns": list(missing_required),
        })

    # ------------------------------------------------------------------
    # Load graph from S3 (YAML ONLY)
    # ------------------------------------------------------------------
    try:
        with tempfile.NamedTemporaryFile(suffix=".yml", delete=True) as tmp_graph_file:
            local_graph_path = tmp_graph_file.name

            logger.info(
                f"[{job_id}] Downloading pipeline graph (YAML) from {graph_uri} "
                f"to temporary file {local_graph_path}"
            )
            s3.get(graph_uri, local_graph_path)

            logger.info(f"[{job_id}] Loading ExecutableGraph from YAML")
            graph = ExecutableGraph.from_yaml(local_graph_path)

    except FileNotFoundError:
        logger.error(f"[{job_id}] Graph file not found at S3 URI: {graph_uri}")
        raise HTTPException(status_code=500, detail=f"Could not load pipeline graph from {graph_uri}")
    except Exception as e:
        logger.error(f"[{job_id}] Failed to download or load graph: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize pipeline: {e}")

    # Determine original filename and set up paths
    original_filename = extract_filename(input_path)
    logger.info(f"[{job_id}] New job submission started for input path: {input_path}")
    
    # Dispatch the FastAPI Background task with the input path
    try:
        job_start = time.time()
        logger.debug(f"[{job_id}] Dispatching FastAPI background task with input: {input_path}")
        # Use FastAPI's BackgroundTasks instead of Celery
        background_tasks.add_task(
            run_graph_job,
            job_id=job_id,
            input_s3_uri=input_path,
            graph=graph,
            column_mapping=mapping_dict,
            required_columns=list(required_cols),
            task_store=job_status_store
        )
        
        task_init_duration = (time.time() - job_start) * 1000  # ms
        
        log_event(
            "task_submitted", 
            job_id=job_id, 
            duration_ms=task_init_duration,
            input_path=input_path,
            pipeline_name=pipeline_name,
            column_mapping=mapping_dict
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
        "pipeline_name": pipeline_name,
        "column_mapping": mapping_dict,
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
    logger.debug(f"[{job_id}] Status check requested using in-memory store.")
    
    job_info = job_status_store.get(job_id)

    if not job_info:
        logger.warning(f"[{job_id}] Job ID not found in job_status_store")
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found. It may not have been submitted or has been cleared.")

    current_status_from_store: Status = job_info.get("status", Status.UNKNOWN) # Default to UNKNOWN if status somehow missing
    
    result_s3_uri: str | None = None
    download_url: str | None = None
    user_message: str = f"Job '{job_id}' status is {str(current_status_from_store).split('.')[-1]}." # Default message
    error_info_details: str | None = None

    if current_status_from_store == Status.SUCCESS:
        user_friendly_status = "COMPLETED"
        result_s3_uri = job_info.get("result")
        
        if result_s3_uri:
            try:
                # Generate a pre-signed URL for the result archive
                bucket_name, key = parse_s3_uri(result_s3_uri)
                if bucket_name and key:
                    download_url = s3_client.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': bucket_name, 'Key': key},
                        ExpiresIn=3600  # URL expires in 1 hour
                    )
                    user_message = "Job completed successfully. Click the link to download your results."
                    logger.info(f"[{job_id}] Generated pre-signed URL for {result_s3_uri}")
                else:
                    user_message = "Job completed, but result path is invalid."
                    logger.warning(f"[{job_id}] Could not parse bucket/key from result URI: {result_s3_uri}")
            except Exception as e:
                user_message = "Job completed, but failed to generate download link."
                logger.error(f"[{job_id}] Failed to generate pre-signed URL: {e}", exc_info=True)
        else:
            user_message = "Job completed successfully, but the result path is missing."
            logger.warning(f"[{job_id}] Job status is SUCCESS but no result path found.")
            
    elif current_status_from_store == Status.FAILED:
        user_friendly_status = "FAILED"
        error_info_details = job_info.get("error")
        user_message = f"Job '{job_id}' failed."
        if error_info_details:
            logger.error(f"[{job_id}] Job failed: {error_info_details}")
        else:
            logger.warning(f"[{job_id}] Job status is FAILED but no error details found.")
            user_message += " However, specific error details are missing."

    elif current_status_from_store == Status.RUNNING:
        user_friendly_status = "RUNNING"
        user_message = f"Job '{job_id}' is currently running."
        # Optionally, you could add more details if your worker updates 'job_info' with current step
        # current_step = job_info.get("current_step", "processing")
        # user_message = f"Job '{job_id}' is running: {current_step}."
        logger.info(f"[{job_id}] Job is running.")
        
    elif current_status_from_store == Status.PENDING:
        user_friendly_status = "PENDING"
        user_message = f"Job '{job_id}' is queued and waiting to start."
        logger.info(f"[{job_id}] Job is pending/queued.")
        
    else: # Handles any other Status enum values or if status is not one of the above
        # Uses the default user_message set earlier, which is based on the string representation of the enum
        user_friendly_status = str(current_status_from_store).split('.')[-1].upper() # Example: "Status.PROCESSING" -> "PROCESSING"
        logger.info(f"[{job_id}] Job is in state: {user_friendly_status}")


    total_duration = (time.time() - start_time) * 1000  # ms
    log_event(
        "job_status_check", 
        job_id=job_id,
        duration_ms=total_duration,
        retrieved_status=str(current_status_from_store), # Log the actual enum value string
        user_status=user_friendly_status
    )

    return JobStatusResponse(
        job_id=job_id,
        status=user_friendly_status,
        message=user_message,
        result_s3_uri=result_s3_uri,
        download_url=download_url,
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
    health_keys = health_status.keys()
    
    # Check S3 connectivity if we have S3 configuration
    if S3_OUTPUT_DIR:
        try:
            s3_start = time.time()
            # Check if bucket exists by parsing and checking
            scheme, bucket, _ = parse_path(S3_OUTPUT_DIR)
            if scheme == "s3":
                s3_client.list_buckets()
                health_status["s3"] = "healthy"
        except Exception as e:
            health_status["s3"] = "unhealthy"
            health_status["s3_error"] = str(e)
    
    # Overall health determination
    health_status["overall"] = "healthy" if all(
        health_status[key] == "healthy" for key in health_keys 
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

# ----------------------
# Endpoint: List Pipelines (uses shared util)
# ----------------------

@app.get("/pipelines", summary="List deployed pipelines and their required/optional columns")
async def list_pipelines():
    """Return a mapping of pipeline names to their required and optional columns.

    Response example::

        {
            "example_graph": {
                "required_columns": ["id_unit", "t", "treat", "y", "post"],
                "optional_columns": []
            }
        }
    """

    pipelines_definition = load_deployed_pipelines()
    return pipelines_definition
