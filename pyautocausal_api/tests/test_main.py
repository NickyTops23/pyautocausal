import os
import uuid
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import UploadFile
from pathlib import Path
import io
import json
from botocore.exceptions import NoCredentialsError, ClientError

# Import the FastAPI app
from app.main import app, parse_s3_uri, job_to_celery_map

# Create a TestClient for the FastAPI app
client = TestClient(app)

# Fixture to mock boto3 s3_client
@pytest.fixture
def mock_s3_client():
    with patch('app.main.s3_client') as mock_client:
        yield mock_client

# Fixture to mock Celery task
@pytest.fixture
def mock_celery_task():
    with patch('app.main.run_graph_job.delay', return_value=MagicMock(id='mock-task-id')) as mock_task:
        yield mock_task

# Fixture to provide a sample CSV file for testing
@pytest.fixture
def sample_csv_file():
    content = b"col1,col2,col3\n1,2,3\n4,5,6"
    return io.BytesIO(content)

# Fixture to provide a UUID for testing
@pytest.fixture
def job_id():
    return str(uuid.uuid4())

# Fixture to mock the S3 bucket environment variable - use patch instead of setting env vars
@pytest.fixture
def mock_s3_env():
    # Patch S3_INPUT_BUCKET directly in the app.main module
    with patch('app.main.S3_INPUT_BUCKET', 's3://test-bucket/base/path'):
        with patch('app.main.S3_REGION', 'us-east-2'):
            yield

# Test the parse_s3_uri function
def test_parse_s3_uri():
    # Test with empty input
    assert parse_s3_uri(None) == (None, None)
    assert parse_s3_uri('') == (None, None)
    
    # Test with invalid URI (no s3:// prefix)
    assert parse_s3_uri('invalid-bucket/path') == (None, None)
    
    # Test with valid bucket, no path
    bucket, key = parse_s3_uri('s3://my-bucket')
    assert bucket == 'my-bucket'
    assert key == ''
    
    # Test with valid bucket and path
    bucket, key = parse_s3_uri('s3://my-bucket/path/to/file')
    assert bucket == 'my-bucket'
    assert key == 'path/to/file'
    
    # Test with trailing slash
    bucket, key = parse_s3_uri('s3://my-bucket/')
    assert bucket == 'my-bucket'
    assert key == ''

# Test the root endpoint
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "PyAutoCausal API is running" in response.json()["message"]

# Test job submission with missing S3 bucket config
def test_submit_job_missing_config():
    with patch('app.main.S3_INPUT_BUCKET', None):
        response = client.post(
            "/jobs",
            files={"file": ("test.csv", io.BytesIO(b"test"), "text/csv")}
        )
        assert response.status_code == 500
        assert "Server configuration error" in response.json()["detail"]
        assert "S3 input bucket not specified" in response.json()["detail"]

# Test job submission with invalid S3 URI format
def test_submit_job_invalid_s3_uri():
    with patch('app.main.S3_INPUT_BUCKET', 'invalid-uri'):
        response = client.post(
            "/jobs",
            files={"file": ("test.csv", io.BytesIO(b"test"), "text/csv")}
        )
        assert response.status_code == 500
        assert "Invalid S3 input bucket URI format" in response.json()["detail"]

# Test successful job submission
@patch('uuid.uuid4', return_value=MagicMock(spec=uuid.UUID, __str__=lambda _: "test-uuid"))
def test_submit_job_success(mock_uuid, mock_s3_client, mock_celery_task, mock_s3_env, sample_csv_file):
    # Setup the request and mock objects
    response = client.post(
        "/jobs",
        files={"file": ("test.csv", sample_csv_file, "text/csv")}
    )
    
    # Check response
    assert response.status_code == 202
    assert response.json()["job_id"] == "test-uuid"
    assert "/jobs/test-uuid" in response.json()["status_url"]
    assert "Job submitted successfully" in response.json()["message"]
    
    # Verify S3 upload was called with correct parameters
    mock_s3_client.upload_fileobj.assert_called_once()
    # The call arguments can be verified further if needed
    
    # Verify Celery task was called with correct parameters
    mock_celery_task.assert_called_once()
    args = mock_celery_task.call_args[1]
    assert args["job_id"] == "test-uuid"
    assert "test.csv" in args["original_filename"]
    assert args["input_s3_uri"].startswith("s3://")
    
    # Verify job_to_celery_map was updated
    assert "test-uuid" in job_to_celery_map
    assert job_to_celery_map["test-uuid"]["celery_task_id"] == "mock-task-id"

# Test S3 upload failure
def test_submit_job_s3_upload_failure(mock_s3_client, mock_s3_env):
    # Make S3 upload fail with NoCredentialsError
    mock_s3_client.upload_fileobj.side_effect = NoCredentialsError()
    
    response = client.post(
        "/jobs",
        files={"file": ("test.csv", io.BytesIO(b"test"), "text/csv")}
    )
    
    assert response.status_code == 500
    assert "AWS credentials not found" in response.json()["detail"]

# Test Celery task submission failure
def test_submit_job_celery_failure(mock_s3_client, mock_s3_env):
    # Make Celery task submission fail
    with patch('app.main.run_graph_job.delay', side_effect=Exception("Celery connection error")):
        response = client.post(
            "/jobs",
            files={"file": ("test.csv", io.BytesIO(b"test"), "text/csv")}
        )
        
        assert response.status_code == 503
        assert "Job queueing service unavailable" in response.json()["detail"]
        
        # Check that S3 cleanup was attempted
        mock_s3_client.delete_object.assert_called_once()

# Test job status retrieval for non-existent job
def test_get_job_status_not_found():
    response = client.get("/jobs/non-existent-job")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]

# Test job status retrieval for job in PENDING state
def test_get_job_status_pending(job_id):
    # Create a mock AsyncResult
    mock_result = MagicMock()
    mock_result.status = "PENDING"
    mock_result.successful.return_value = False
    mock_result.failed.return_value = False
    mock_result.result = None
    mock_result.state = "PENDING"
    
    # Add job to job_to_celery_map
    job_to_celery_map[job_id] = {
        "celery_task_id": "mock-task-id",
        "original_filename": "test.csv",
        "input_s3_uri": "s3://test-bucket/test.csv"
    }
    
    with patch('app.main.AsyncResult', return_value=mock_result):
        response = client.get(f"/jobs/{job_id}")
        
        assert response.status_code == 200
        assert response.json()["job_id"] == job_id
        assert response.json()["status"] == "PENDING"
        assert "queued and waiting" in response.json()["message"]
        assert response.json()["result_path"] is None
        assert response.json()["error_details"] is None
        
    # Clean up
    job_to_celery_map.pop(job_id, None)

# Test job status retrieval for job in SUCCESS state
def test_get_job_status_success(job_id):
    # Create a mock AsyncResult
    mock_result = MagicMock()
    mock_result.status = "SUCCESS"
    mock_result.successful.return_value = True
    mock_result.failed.return_value = False
    mock_result.result = "s3://output-bucket/outputs/job-id/"
    
    # Add job to job_to_celery_map
    job_to_celery_map[job_id] = {
        "celery_task_id": "mock-task-id",
        "original_filename": "test.csv",
        "input_s3_uri": "s3://test-bucket/test.csv"
    }
    
    with patch('app.main.AsyncResult', return_value=mock_result):
        response = client.get(f"/jobs/{job_id}")
        
        assert response.status_code == 200
        assert response.json()["job_id"] == job_id
        assert response.json()["status"] == "COMPLETED"
        assert "completed successfully" in response.json()["message"]
        assert response.json()["result_path"] == "s3://output-bucket/outputs/job-id/"
        assert response.json()["error_details"] is None
        
    # Clean up
    job_to_celery_map.pop(job_id, None)

# Test job status retrieval for job in FAILURE state
def test_get_job_status_failure(job_id):
    # Create a mock exception
    mock_exception = ValueError("Test error")
    
    # Create a mock AsyncResult
    mock_result = MagicMock()
    mock_result.status = "FAILURE"
    mock_result.successful.return_value = False
    mock_result.failed.return_value = True
    mock_result.result = mock_exception
    
    # Add job to job_to_celery_map
    job_to_celery_map[job_id] = {
        "celery_task_id": "mock-task-id",
        "original_filename": "test.csv",
        "input_s3_uri": "s3://test-bucket/test.csv"
    }
    
    with patch('app.main.AsyncResult', return_value=mock_result):
        response = client.get(f"/jobs/{job_id}")
        
        assert response.status_code == 200
        assert response.json()["job_id"] == job_id
        assert response.json()["status"] == "FAILED"
        assert "failed" in response.json()["message"]
        assert response.json()["result_path"] is None
        assert "ValueError: Test error" in response.json()["error_details"]
        
    # Clean up
    job_to_celery_map.pop(job_id, None)

# Test job status retrieval for job in custom PROCESSING state
def test_get_job_status_processing(job_id):
    # Create a mock AsyncResult
    mock_result = MagicMock()
    mock_result.status = "PROCESSING"
    mock_result.state = "PROCESSING"
    mock_result.successful.return_value = False
    mock_result.failed.return_value = False
    mock_result.info = {"message": "Loading data from local copy"}
    
    # Add job to job_to_celery_map
    job_to_celery_map[job_id] = {
        "celery_task_id": "mock-task-id",
        "original_filename": "test.csv",
        "input_s3_uri": "s3://test-bucket/test.csv"
    }
    
    with patch('app.main.AsyncResult', return_value=mock_result):
        response = client.get(f"/jobs/{job_id}")
        
        assert response.status_code == 200
        assert response.json()["job_id"] == job_id
        assert response.json()["status"] == "PROCESSING"
        assert "Loading data from local copy" in response.json()["message"]
        
    # Clean up
    job_to_celery_map.pop(job_id, None) 