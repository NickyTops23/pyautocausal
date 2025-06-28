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
import io

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
def test_run_graph_job_missing_output_bucket(mock_task_store, job_id, mock_graph):
    with patch('app.worker.S3_OUTPUT_DIR', None):
        mock_task_store[job_id] = {
            "status": Status.PENDING,
            "original_filename": "test.csv",
            "input_path": "s3://input-bucket/file.csv",
            "result": None,
            "error": None
        }
        
        with pytest.raises(ValueError) as excinfo:
            worker.run_graph_job(job_id, "s3://input-bucket/file.csv", mock_graph, {}, [], mock_task_store)
        
        assert "S3 output bucket not specified" in str(excinfo.value)
        assert mock_task_store[job_id]["status"] == Status.FAILED
        assert "S3 output bucket not specified" in mock_task_store[job_id]["error"]

# Test _run_graph_job with invalid S3_OUTPUT_BUCKET format
def test_run_graph_job_invalid_output_bucket_format(mock_task_store, job_id, mock_graph):
    with patch('app.worker.S3_OUTPUT_DIR', 'invalid-format'):
        mock_task_store[job_id] = {
            "status": Status.PENDING,
            "original_filename": "test.csv",
            "input_path": "s3://input-bucket/file.csv",
            "result": None,
            "error": None
        }
        
        with pytest.raises(ValueError) as excinfo:
            worker.run_graph_job(job_id, "s3://input-bucket/file.csv", mock_graph, {}, [], mock_task_store)
        
        assert "Invalid S3 output bucket URI format" in str(excinfo.value)
        assert mock_task_store[job_id]["status"] == Status.FAILED
        assert "Invalid S3 output bucket URI format" in mock_task_store[job_id]["error"]

# Test successful execution (end-to-end)
@patch('shutil.rmtree')  # Mock rmtree to avoid actual directory removal
@patch('shutil.make_archive')
@patch('app.worker.get_s3_fs')
def test_run_graph_job_success(mock_get_s3, mock_make_archive, mock_rmtree, mock_task_store, mock_s3_client, mock_s3_env, 
                              mock_graph, mock_notebook_exporter, temp_dir, job_id):
    
    # Configure the mock for get_s3_fs
    mock_s3fs = MagicMock()
    mock_get_s3.return_value = mock_s3fs
    
    # Patch the WORKER_TEMP_DIR to use our temp directory
    with patch('app.worker.WORKER_TEMP_DIR', Path(str(temp_dir))):
        # We no longer need to patch 'app.worker.s3' here
        with patch('app.worker.load_deployed_pipelines') as mock_load_pipelines:
            mock_load_pipelines.return_value = {
                "example_graph": {
                    "required_columns": ["id_unit", "t", "treat", "y", "post"],
                    "optional_columns": [],
                    "graph_uri": "s3://test-bucket/example_graph.pkl"
                }
            }
                
        # Create test directories
        output_dir = temp_dir.join(f"output/{job_id}")
        output_dir.ensure_dir()
        
        mock_task_store[job_id] = {
            "status": Status.PENDING,
            "original_filename": "test.csv",
            "input_path": "s3://input-bucket/test.csv",
            "result": None,
            "error": None
        }
        
        # Create a test CSV data
        data = generate_mock_data(n_units=200, n_periods=2, n_treated=50)
        
        # Mock the path to the created archive
        archive_name = f"pyautocausal_results_{job_id}.zip"
        archive_path = temp_dir.join("output").join(archive_name)
        archive_path.write("zip_content") # create a dummy zip file
        mock_make_archive.return_value = str(archive_path)
            
        # Mock the s3fs open method to simulate reading a CSV from S3
        mock_s3fs.open.return_value.__enter__.return_value = io.StringIO(data.to_csv(index=False))

        # Run the task with new signature
        column_mapping = {"id_unit": "id_unit", "t": "t", "treat": "treat", "y": "y", "post": "post"}
        required_columns = list(column_mapping.values())
        worker.run_graph_job(job_id, "s3://input-bucket/test.csv", mock_graph, column_mapping, required_columns, mock_task_store)
        
        # Check the task store was updated correctly
        assert mock_task_store[job_id]["status"] == Status.SUCCESS
        final_result_uri = mock_task_store[job_id]["result"]
        assert final_result_uri.startswith("s3://test-output-bucket/output/path")
        assert final_result_uri.endswith(f"/{job_id}/{archive_name}")
        
        # Verify graph was configured and fit
        mock_graph.configure_runtime.assert_called_once()
        mock_graph.fit.assert_called_once()
        
        # Verify notebook exporter was called
        mock_notebook_exporter.export_notebook.assert_called_once()

        # Verify that make_archive was called correctly
        mock_make_archive.assert_called_once()
        
        # Verify s3fs.put was called to upload the archive
        mock_s3fs.put.assert_called_once_with(str(archive_path), final_result_uri)
        
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