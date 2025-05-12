import pytest
import time
import os
import uuid
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Set memory backend for testing
os.environ['CELERY_BROKER_URL'] = 'memory://'
os.environ['CELERY_RESULT_BACKEND'] = 'memory://'
os.environ['S3_OUTPUT_BUCKET'] = 's3://test-bucket/outputs/'
os.environ['AWS_REGION'] = 'us-east-2'

# print the current working directory
print(os.getcwd())

# Import the FastAPI app and celery app AFTER setting environment variables
from app.main import app
from app.worker import celery_app, run_graph_job
from fastapi.testclient import TestClient

# Create test client
client = TestClient(app)

# Fixture for S3 mocking
@pytest.fixture
def mock_s3():
    with patch('app.worker.s3_client') as mock_s3:
        # Setup mock responses
        mock_s3.head_object.return_value = {'ContentLength': 1024}
        
        # Mock download_file to create a test CSV
        def fake_download(bucket, key, path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write("col1,col2,col3\n1,2,3\n4,5,6")
        mock_s3.download_file.side_effect = fake_download
        
        # Mock upload_file to do nothing
        mock_s3.upload_file.return_value = None
        
        yield mock_s3

# Patch temporary directory
@pytest.fixture
def mock_temp_dir(tmpdir):
    with patch('app.worker.WORKER_TEMP_DIR', Path(tmpdir)):
        yield tmpdir

# Test synchronous execution of a Celery task
def test_sync_celery_execution(mock_s3, mock_temp_dir):
    """Test that a Celery task executes synchronously with memory backend"""
    
    # Create a unique job ID
    job_id = str(uuid.uuid4())
    s3_uri = "s3://test-bucket/test.csv"
    filename = "test.csv"
    
    # Execute the task synchronously (doesn't use workers)
    # This is key - with apply() the task runs in the current process
    task_result = run_graph_job.apply(args=[job_id, s3_uri, filename])
    
    # Assert the task completed successfully
    assert task_result.successful()
    
    # Verify the result contains the S3 output URI
    result = task_result.result
    assert result.startswith("s3://")
    assert "test-bucket" in result
    assert job_id in result
    
    # Verify S3 calls were made
    mock_s3.head_object.assert_called_once()
    mock_s3.download_file.assert_called_once()
    assert mock_s3.upload_file.call_count > 0

# Test the full API flow with in-memory Celery
def test_api_flow_with_memory_celery(mock_s3, mock_temp_dir):
    """Test the entire API flow using the memory Celery backend"""
    
    # 1. Submit a job through the API
    response = client.post(
        "/jobs/s3",
        json={"s3_uri": "s3://test-bucket/test.csv"}
    )
    
    # Verify the response
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    
    # Give the task a moment to be processed
    # Not strictly necessary with memory backend and apply(), but included for completeness
    time.sleep(1)
    
    # 2. Check the job status
    status_response = client.get(f"/jobs/{job_id}")
    
    # Print details for debugging
    print(f"Job Status Response: {status_response.json()}")
    
    # Verify the job was processed
    assert status_response.status_code == 200
    assert status_response.json()["status"] != "PENDING"

# Test with breakpoint for debugging
def test_with_debugger(mock_s3, mock_temp_dir):
    """This test includes a breakpoint for debugging with pdb"""
    
    # Setup test data
    job_id = str(uuid.uuid4())
    s3_uri = "s3://test-bucket/test.csv"
    filename = "test.csv"
    
    # Add a breakpoint here for debugging
    # import pdb; pdb.set_trace()
    
    # Submit task
    task = run_graph_job.apply(args=[job_id, s3_uri, filename])
    
    # Verify result
    assert task.successful()
    result = task.result
    assert "s3://" in result

if __name__ == "__main__":
    # This allows running the file directly for debugging
    # pytest will skip this when running with pytest command
    
    # Set up basic test environment
    os.environ['CELERY_BROKER_URL'] = 'memory://'
    os.environ['CELERY_RESULT_BACKEND'] = 'memory://'
    os.environ['S3_OUTPUT_BUCKET'] = 's3://test-bucket/outputs/'
    os.environ['AWS_REGION'] = 'us-east-2'
    
    # Import the worker module
    from app.worker import celery_app, run_graph_job
    
    # Debug print celery configuration
    print("Celery Configuration:")
    print(f"Broker URL: {celery_app.conf.broker_url}")
    print(f"Result Backend: {celery_app.conf.result_backend}")
    
    # Create a test job
    job_id = str(uuid.uuid4())
    s3_uri = "s3://test-bucket/test.csv"
    filename = "test.csv"
    
    # Set up temporary directory
    import tempfile
    temp_dir = tempfile.mkdtemp()
    os.environ['WORKER_TEMP_DIR'] = temp_dir
    
    # Mock S3 client
    from unittest.mock import patch, MagicMock
    with patch('app.worker.s3_client') as mock_s3:
        # Setup mock for head_object
        mock_s3.head_object.return_value = {'ContentLength': 1024}
        
        # Setup mock for download_file
        def fake_download(bucket, key, path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write("col1,col2,col3\n1,2,3\n4,5,6")
        mock_s3.download_file.side_effect = fake_download
        
        # Setup mock for upload_file
        mock_s3.upload_file.return_value = None
        
        # Add a breakpoint before task execution for debugging
        import pdb; pdb.set_trace()
        
        # Execute task
        print("Executing task...")
        task = run_graph_job.apply(args=[job_id, s3_uri, filename])
        
        # Print result
        print(f"Task state: {task.state}")
        print(f"Task result: {task.result}")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
