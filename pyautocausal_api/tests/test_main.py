import os
import uuid
from unittest.mock import patch, MagicMock, AsyncMock
from starlette.testclient import TestClient
from fastapi import UploadFile, BackgroundTasks
from pathlib import Path
import io
import json
from botocore.exceptions import NoCredentialsError, ClientError
import pytest

# Patch httpx.Client to accept 'app' argument for compatibility with starlette TestClient and fastapi TestClient
import httpx
from httpx import ASGITransport

_orig_client_init = httpx.Client.__init__

def _patched_client_init(self, *args, **kwargs):
    if 'app' in kwargs and 'transport' not in kwargs:
        app_ = kwargs.pop('app')
        kwargs['transport'] = ASGITransport(app=app_)
    return _orig_client_init(self, *args, **kwargs)

httpx.Client.__init__ = _patched_client_init

# Import the FastAPI app and other necessary components
from app.main import app, parse_s3_uri, job_status_store
from app.utils.file_io import Status
from app.utils.pipeline_registry import load_deployed_pipelines
from pyautocausal.orchestration.graph import ExecutableGraph

# Create a TestClient for the FastAPI app
client = TestClient(app)

# Fixture to mock boto3 s3_client
@pytest.fixture
def mock_s3_client():
    with patch('app.utils.file_io.s3_client') as mock_client:
        mock_client.list_buckets.return_value = {"Buckets": [{"Name": "test-bucket"}]}
        yield mock_client

# Fixture to provide a sample CSV file for testing
@pytest.fixture
def sample_csv_file():
    content = b"col1,col2,col3\n1,2,3\n4,5,6"
    return io.BytesIO(content)

# Fixture to provide a UUID for testing
@pytest.fixture
def job_id_fixture():
    return str(uuid.uuid4())

# Fixture to mock the S3 environment variables
@pytest.fixture
def mock_s3_env_vars():
    with patch.dict(os.environ, {
        # "S3_BUCKET_INPUT": "s3://test-input-bucket/inputs", # This might still be useful if other modules use it
        "S3_BUCKET_OUTPUT": "s3://test-output-bucket/outputs",
        "AWS_REGION": "us-east-1"
    }):
        with patch('app.main.S3_OUTPUT_DIR', "s3://test-output-bucket/outputs"):
            with patch('app.main.S3_REGION', "us-east-1"):
                yield

# Test the parse_s3_uri function
def test_parse_s3_uri():
    assert parse_s3_uri(None) == (None, None)
    assert parse_s3_uri('') == (None, None)
    assert parse_s3_uri('invalid-bucket/path') == (None, None)
    bucket, key = parse_s3_uri('s3://my-bucket')
    assert bucket == 'my-bucket'
    assert key == ''
    bucket, key = parse_s3_uri('s3://my-bucket/path/to/file')
    assert bucket == 'my-bucket'
    assert key == 'path/to/file'
    bucket, key = parse_s3_uri('s3://my-bucket/')
    assert bucket == 'my-bucket'
    assert key == ''

# Test the root endpoint
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "PyAutoCausal API is running" in response.json()["message"]

# Test job submission with missing S3_INPUT_DIR config
# This test is no longer relevant as S3_INPUT_DIR check for intermediate upload is removed from submit_job.
# @patch('app.main.read_from')
# def test_submit_job_missing_input_dir_config(mock_read_from_main):
#     mock_read_from_main.return_value = b"dummy file content"
#     with patch('app.main.S3_INPUT_DIR', None): 
#         response = client.post(
#             "/jobs",
#             data={"input_path": "s3://some-bucket/some-file.csv"}
#         )
#         assert response.status_code == 500
#         assert "Server configuration error" in response.json()["detail"]
#         assert "S3 input bucket not specified" in response.json()["detail"]

# Test S3 upload failure (mocking write_to)
# This test is no longer relevant as the endpoint doesn't do the intermediate write.
# @patch('uuid.uuid4')
# @patch('app.main.read_from')
# @patch('app.main.write_to')
# def test_submit_job_s3_upload_failure(
#     mock_write_to, mock_read_from, mock_uuid_func,
#     mock_s3_env_vars, job_id_fixture
# ):
#     mock_uuid_func.return_value = MagicMock(spec=uuid.UUID, __str__=lambda _: job_id_fixture)
#     mock_read_from.return_value = b"file,content"
#     mock_write_to.side_effect = NoCredentialsError()
# 
#     job_status_store.clear()
#     
#     response = client.post(
#         "/jobs",
#         data={"input_path": "s3://test-bucket/source/test.csv"}
#     )
#     
#     assert response.status_code == 500
#     assert "Error during file upload" in response.json()["detail"]

# Test Background Task submission failure
@patch('uuid.uuid4')
@patch.object(BackgroundTasks, 'add_task')
def test_submit_job_validation_failure(
    mock_add_task, mock_uuid_func,
    mock_s3_env_vars, job_id_fixture
):
    mock_uuid_func.return_value = MagicMock(spec=uuid.UUID, __str__=lambda _: job_id_fixture)

    job_status_store.clear()

    response = client.post(
        "/jobs",
        json={
            "input_path": "s3://test-bucket/source/test.csv",
            "pipeline_name": list(load_deployed_pipelines().keys())[0],
            "column_mapping": {},
        }
    )
        
    assert response.status_code == 422
    assert "Missing required column mappings" in response.json()["detail"]["error"]
    
    assert job_id_fixture not in job_status_store

# Test actual Background Task submission failure
@patch('app.main.ExecutableGraph.load')
@patch('app.main.s3.get')
@patch('uuid.uuid4')
@patch.object(BackgroundTasks, 'add_task')
def test_submit_job_task_submission_failure(
    mock_add_task, mock_uuid_func, mock_s3_get, mock_graph_load,
    mock_s3_env_vars, job_id_fixture
):
    mock_uuid_func.return_value = MagicMock(spec=uuid.UUID, __str__=lambda _: job_id_fixture)
    mock_add_task.side_effect = Exception("Queueing service error")
    mock_graph_load.return_value = MagicMock(spec=ExecutableGraph)

    job_status_store.clear()

    # Use valid mapping to pass validation
    pipelines = load_deployed_pipelines()
    pipeline_name = list(pipelines.keys())[0]
    required_cols = pipelines[pipeline_name]["required_columns"]
    mapping = {f"col{i}": col for i, col in enumerate(required_cols)}

    response = client.post(
        "/jobs",
        json={
            "input_path": "s3://test-bucket/source/test.csv",
            "pipeline_name": pipeline_name,
            "column_mapping": mapping,
        }
    )
        
    assert response.status_code == 503
    assert "Job queueing service unavailable" in response.json()["detail"]
    assert "Queueing service error" in response.json()["detail"]
    
    assert job_id_fixture not in job_status_store

# Test job status retrieval for non-existent job
def test_get_job_status_not_found(job_id_fixture):
    job_status_store.clear()
    response = client.get(f"/jobs/{job_id_fixture}")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]

# Test job status retrieval for job in PENDING state
def test_get_job_status_pending(job_id_fixture):
    job_status_store.clear()
    job_status_store[job_id_fixture] = {
        "status": Status.PENDING,
        "original_filename": "test.csv",
        "input_path": "s3://test-bucket/test.csv",
        "result": None,
        "error": None
    }
    
    response = client.get(f"/jobs/{job_id_fixture}")
        
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["job_id"] == job_id_fixture
    assert json_response["status"] == "PENDING"
    assert "queued and waiting" in json_response["message"]
    assert json_response["result_path"] is None
    assert json_response["error_details"] is None
        
    job_status_store.pop(job_id_fixture, None)

# Test job status retrieval for job in RUNNING state
def test_get_job_status_running(job_id_fixture):
    job_status_store.clear()
    job_status_store[job_id_fixture] = {
        "status": Status.RUNNING,
        "original_filename": "test.csv",
        "input_path": "s3://test-bucket/test.csv",
        "result": None,
        "error": None
    }
    
    response = client.get(f"/jobs/{job_id_fixture}")
        
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["job_id"] == job_id_fixture
    assert json_response["status"] == "RUNNING"
    assert "currently running" in json_response["message"]
    assert json_response["result_path"] is None
    assert json_response["error_details"] is None
        
    job_status_store.pop(job_id_fixture, None)

# Test job status retrieval for job in SUCCESS state
def test_get_job_status_success(job_id_fixture):
    job_status_store.clear()
    output_s3_path = "s3://output-bucket/outputs/test-job-id/"
    job_status_store[job_id_fixture] = {
        "status": Status.SUCCESS,
        "original_filename": "test.csv",
        "input_path": "s3://test-bucket/test.csv",
        "result": output_s3_path,
        "error": None
    }
    
    response = client.get(f"/jobs/{job_id_fixture}")
        
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["job_id"] == job_id_fixture
    assert json_response["status"] == "COMPLETED"
    assert "completed successfully" in json_response["message"]
    assert json_response["result_path"] == output_s3_path
    assert json_response["error_details"] is None
        
    job_status_store.pop(job_id_fixture, None)

# Test job status retrieval for job in FAILURE state
def test_get_job_status_failure(job_id_fixture):
    job_status_store.clear()
    error_message = "ValueError: Something went wrong during processing"
    job_status_store[job_id_fixture] = {
        "status": Status.FAILED,
        "original_filename": "test.csv",
        "input_path": "s3://test-bucket/test.csv",
        "result": None,
        "error": error_message
    }
    
    response = client.get(f"/jobs/{job_id_fixture}")
        
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["job_id"] == job_id_fixture
    assert json_response["status"] == "FAILED"
    assert "failed" in json_response["message"]
    assert json_response["result_path"] is None
    assert error_message in json_response["error_details"]
        
    job_status_store.pop(job_id_fixture, None)

# # Test for health check endpoint
# def test_health_check(mock_s3_client, mock_s3_env_vars):

#     # Mock s3_client.list_buckets() as used in health check
#     mock_s3_client.list_buckets.return_value = {"Buckets": [{"Name": "test-output-bucket"}]}

#     response = client.get("/health")
#     assert response.status_code == 200
#     health_data = response.json()
#     assert health_data["api"] == "healthy"
#     assert health_data["s3"] == "healthy"
#     assert health_data["overall"] == "healthy"

# Test for a case where input_path is not provided in submit_job
def test_submit_job_missing_input_path(mock_s3_env_vars):
    response = client.post(
        "/jobs",
        json={
            "pipeline_name": list(load_deployed_pipelines().keys())[0],
            "column_mapping": {},
        }
    )
    assert response.status_code == 422
    details = response.json().get("detail")
    assert any("input_path" in err.get("loc", []) and "field required" in err.get("msg", "").lower() for err in details) 

# New Test for successful job submission with graph loading
@patch('uuid.uuid4')
@patch.object(BackgroundTasks, 'add_task')
@patch('app.main.s3.get')
@patch('app.main.ExecutableGraph.load')
def test_submit_job_success_with_graph_loading(
    mock_graph_load, mock_s3_get, mock_add_task, mock_uuid_func,
    mock_s3_env_vars, job_id_fixture
):
    # Mocking
    mock_uuid_func.return_value = MagicMock(spec=uuid.UUID, __str__=lambda _: job_id_fixture)
    mock_graph = MagicMock(spec=ExecutableGraph)
    mock_graph_load.return_value = mock_graph
    
    job_status_store.clear()

    input_s3_path = "s3://test-bucket/source/test.csv"
    pipelines = load_deployed_pipelines()
    pipeline_name = list(pipelines.keys())[0]
    required_cols = pipelines[pipeline_name]["required_columns"]
    mapping = {f"user_col_{i}": col for i, col in enumerate(required_cols)}
    
    # Send request
    response = client.post(
        "/jobs",
        json={
            "input_path": input_s3_path,
            "pipeline_name": pipeline_name,
            "column_mapping": mapping,
        }
    )
    
    # Assertions for the response
    assert response.status_code == 202
    json_response = response.json()
    assert json_response["job_id"] == job_id_fixture
    assert f"/jobs/{job_id_fixture}" in json_response["status_url"]
    
    # Assert that S3 download and graph loading were called
    mock_s3_get.assert_called_once()
    mock_graph_load.assert_called_once()
    
    # Assert that the background task was called with the correct parameters
    mock_add_task.assert_called_once()
    args = mock_add_task.call_args[1]
    assert args["job_id"] == job_id_fixture
    assert args["input_s3_uri"] == input_s3_path
    assert args["graph"] is mock_graph # Check that the graph object is passed
    assert args["column_mapping"] == mapping
    assert sorted(args["required_columns"]) == sorted(required_cols)
    assert args["task_store"] is job_status_store
    
    # Assertions for the job store
    assert job_id_fixture in job_status_store
    job_info = job_status_store[job_id_fixture]
    assert job_info["status"] == Status.PENDING
    assert job_info["original_filename"] == "test.csv"
    assert job_info["input_path"] == input_s3_path
    assert job_info["pipeline_name"] == pipeline_name
    assert job_info["result"] is None
    assert job_info["error"] is None