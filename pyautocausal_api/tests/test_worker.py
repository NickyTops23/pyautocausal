import os
from pyautocausal.pipelines.mock_data import generate_mock_data
import pytest
import uuid
import pandas as pd
from pathlib import Path
import shutil
from unittest.mock import patch, MagicMock, Mock
from botocore.exceptions import NoCredentialsError, ClientError
import time

# Import the worker functions
from app import worker
from app.worker import parse_s3_uri
from app.utils.file_io import Status

# Fixture to provide a UUID for testing
@pytest.fixture
def job_id():
    return str(uuid.uuid4())

# Fixture to mock task store
@pytest.fixture
def mock_task_store():
    return {}

# Fixture to mock S3 client
@pytest.fixture
def mock_s3_client():
    with patch('app.worker.s3_client') as mock_client:
        yield mock_client

# Fixture to mock boto3
@pytest.fixture
def mock_boto3():
    with patch('app.worker.boto3') as mock_boto:
        mock_s3_client = MagicMock()
        mock_boto.client.return_value = mock_s3_client
        yield mock_boto

# Fixture to mock S3 environment variables
@pytest.fixture
def mock_s3_env():
    # Create a proper environment with the required settings
    with patch.dict('os.environ', {
        'S3_OUTPUT_BUCKET': 's3://test-output-bucket/output/path',
        'AWS_REGION': 'us-east-2'
    }):
        # Patch the values directly in the worker module to ensure they're properly set
        with patch('app.worker.S3_OUTPUT_DIR', 's3://test-output-bucket/output/path'):
            with patch('app.worker.S3_REGION', 'us-east-2'):
                yield

# Fixture to create a temporary directory structure for testing
@pytest.fixture
def temp_dir(tmpdir):
    # Create temp directories
    temp_job_dir = tmpdir.mkdir("job_id")
    temp_input_dir = temp_job_dir.mkdir("input")
    temp_output_dir = temp_job_dir.mkdir("output")
    
    # Create a sample CSV file
    csv_content = "col1,col2,col3\n1,2,3\n4,5,6"
    csv_file = temp_input_dir.join("test.csv")
    csv_file.write(csv_content)
    
    yield tmpdir
    
    # Cleanup is handled automatically by pytest

# Fixture to mock the graph and NotebookExporter
@pytest.fixture
def mock_graph():
    # Mock ExecutableGraph.load and the graph instance
    with patch('pyautocausal.orchestration.graph.ExecutableGraph.load') as mock_load:
        mock_graph = MagicMock()
        mock_load.return_value = mock_graph
        yield mock_graph

@pytest.fixture
def mock_notebook_exporter():
    with patch('app.worker.NotebookExporter') as mock_exporter_class:
        mock_exporter = MagicMock()
        mock_exporter_class.return_value = mock_exporter
        yield mock_exporter

# Test parse_s3_uri function
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

# Test _run_graph_job missing S3_OUTPUT_BUCKET
def test_run_graph_job_missing_output_bucket(mock_task_store, job_id):
    with patch('app.worker.S3_OUTPUT_DIR', None):
        mock_task_store[job_id] = {
            "status": Status.PENDING,
            "original_filename": "test.csv",
            "input_path": "s3://input-bucket/file.csv",
            "result": None,
            "error": None
        }
        
        with pytest.raises(ValueError) as excinfo:
            worker.run_graph_job(job_id, "s3://input-bucket/file.csv", "example_graph", {}, mock_task_store)
        
        assert "S3 output bucket not specified" in str(excinfo.value)
        assert mock_task_store[job_id]["status"] == Status.FAILED
        assert "S3 output bucket not specified" in mock_task_store[job_id]["error"]

# Test _run_graph_job with invalid S3_OUTPUT_BUCKET format
def test_run_graph_job_invalid_output_bucket_format(mock_task_store, job_id):
    with patch('app.worker.S3_OUTPUT_DIR', 'invalid-format'):
        mock_task_store[job_id] = {
            "status": Status.PENDING,
            "original_filename": "test.csv",
            "input_path": "s3://input-bucket/file.csv",
            "result": None,
            "error": None
        }
        
        with pytest.raises(ValueError) as excinfo:
            worker.run_graph_job(job_id, "s3://input-bucket/file.csv", "example_graph", {}, mock_task_store)
        
        assert "Invalid S3 output bucket URI format" in str(excinfo.value)
        assert mock_task_store[job_id]["status"] == Status.FAILED
        assert "Invalid S3 output bucket URI format" in mock_task_store[job_id]["error"]

# Test successful execution (end-to-end)
@patch('shutil.rmtree')  # Mock rmtree to avoid actual directory removal
def test_run_graph_job_success(mock_rmtree, mock_task_store, mock_s3_client, mock_s3_env, 
                              mock_graph, mock_notebook_exporter, temp_dir, job_id):
    # Patch the WORKER_TEMP_DIR to use our temp directory
    with patch('app.worker.WORKER_TEMP_DIR', Path(str(temp_dir))):
        # Mock s3fs and pipeline registry
        with patch('app.worker.s3') as mock_s3fs:
            with patch('app.worker.load_deployed_pipelines') as mock_load_pipelines:
                mock_load_pipelines.return_value = {
                    "example_graph": {
                        "required_columns": ["id_unit", "t", "treat", "y", "post"],
                        "optional_columns": [],
                        "graph_uri": "s3://test-bucket/example_graph.pkl"
                    }
                }
                
        # Create test directories and a sample CSV
        input_dir = temp_dir.join(f"/input/{job_id}")
        output_dir = temp_dir.join(f"/output/{job_id}")
        input_dir.ensure_dir()
        output_dir.ensure_dir()
        
        mock_task_store[job_id] = {
            "status": Status.PENDING,
            "original_filename": "test.csv",
            "input_path": "s3://input-bucket/test.csv",
            "result": None,
            "error": None
        }
        
        # Create a test CSV file
        data = generate_mock_data(n_units=2000, n_periods=2, n_treated=500)
        csv_file_path = input_dir.join("test.csv")
        data.to_csv(csv_file_path, index=False)
                    
        # Mock pandas read_csv to return our test data
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = data
            
            # Create a test file in output directory for upload testing
            output_file = output_dir.join("result.csv")
            output_file.write("result data")
            
            # Run the task with new signature
            column_mapping = {"id_unit": "id_unit", "t": "t", "treat": "treat", "y": "y", "post": "post"}
            result = worker.run_graph_job(job_id, "s3://input-bucket/test.csv", "example_graph", column_mapping, mock_task_store)
            
            # Verify the task executed correctly
            assert result is None  # run_graph_job doesn't return anything now
            
            # Check the task store was updated correctly
            assert mock_task_store[job_id]["status"] == Status.SUCCESS
            assert mock_task_store[job_id]["result"].startswith("s3://")
            assert "test-output-bucket" in mock_task_store[job_id]["result"]
            assert job_id in mock_task_store[job_id]["result"]
            
            # Verify graph was configured and fit
            mock_graph.configure_runtime.assert_called_once()
            mock_graph.fit.assert_called_once()
            
            # Verify notebook exporter was called
            mock_notebook_exporter.export_notebook.assert_called_once()
            
            # Verify cleanup was attempted
            assert mock_rmtree.called

# --- Test for _get_file_size ---
def test_get_file_size_success(temp_dir):
    """Test _get_file_size successfully returns file size."""
    file_path = Path(temp_dir.join("test_file.txt"))
    file_content = "Hello, world!"
    file_path.write_text(file_content)
    
    expected_size = len(file_content.encode('utf-8')) # Get byte size
    assert worker._get_file_size(file_path) == expected_size

def test_get_file_size_file_not_found(temp_dir):
    """Test _get_file_size when file is not found."""
    non_existent_file = Path(temp_dir.join("non_existent.txt"))
    assert worker._get_file_size(non_existent_file) == 0 # Returns 0 as per function logic
#