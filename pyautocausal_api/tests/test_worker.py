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
    with patch('app.worker.simple_graph') as mock_simple_graph:
        mock_graph = MagicMock()
        mock_simple_graph.return_value = mock_graph
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
            worker.run_graph_job(job_id, "s3://input-bucket/file.csv", mock_task_store)
        
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
            worker.run_graph_job(job_id, "s3://input-bucket/file.csv", mock_task_store)
        
        assert "Invalid S3 output bucket URI format" in str(excinfo.value)
        assert mock_task_store[job_id]["status"] == Status.FAILED
        assert "Invalid S3 output bucket URI format" in mock_task_store[job_id]["error"]

# Test S3 download failure (NoCredentialsError)
def test_run_graph_job_s3_download_credentials_error(mock_task_store, mock_s3_client, mock_s3_env, job_id):
    # Make S3 download fail with NoCredentialsError
    mock_s3_client.download_file.side_effect = NoCredentialsError()
    
    mock_task_store[job_id] = {
        "status": Status.PENDING,
        "original_filename": "test.csv",
        "input_path": "s3://input-bucket/file.csv",
        "result": None,
        "error": None
    }
    
    with pytest.raises(NoCredentialsError):
        worker.run_graph_job(job_id, "s3://input-bucket/file.csv", mock_task_store)
    
    assert mock_task_store[job_id]["status"] == Status.FAILED
    assert "credentials" in mock_task_store[job_id]["error"].lower()

# Test S3 download failure (ClientError)
def test_run_graph_job_s3_download_client_error(mock_task_store, mock_s3_client, mock_s3_env, job_id):
    # Make S3 download fail with ClientError
    mock_s3_client.download_file.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."}},
        "GetObject"
    )
    
    mock_task_store[job_id] = {
        "status": Status.PENDING,
        "original_filename": "test.csv",
        "input_path": "s3://input-bucket/file.csv",
        "result": None,
        "error": None
    }
    
    with pytest.raises(ClientError):
        worker.run_graph_job(job_id, "s3://input-bucket/file.csv", mock_task_store)
    
    assert mock_task_store[job_id]["status"] == Status.FAILED
    assert "nosuchkey" in mock_task_store[job_id]["error"].lower() or "key does not exist" in mock_task_store[job_id]["error"].lower()

# Test CSV parsing error
def test_run_graph_job_csv_parsing_error(mock_task_store, mock_s3_client, mock_s3_env, temp_dir, job_id):
    # Patch the WORKER_TEMP_DIR to use our temp directory
    with patch('app.worker.WORKER_TEMP_DIR', Path(str(temp_dir))):
        # Create an invalid CSV file
        temp_file = temp_dir.join("job_id/input/invalid.csv")
        temp_file.write("this,is,not,a,valid\ncsv,file\nwith,inconsistent,columns")
        
        mock_task_store[job_id] = {
            "status": Status.PENDING,
            "original_filename": "invalid.csv",
            "input_path": "s3://input-bucket/invalid.csv",
            "result": None,
            "error": None
        }
        
        # Mock successful S3 download
        def mock_download_file(bucket, key, file_path):
            # Instead of actually downloading, we just ensure the target file exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(temp_file), file_path)
        
        mock_s3_client.download_file.side_effect = mock_download_file
        
        # Mock pd.read_csv to raise an exception
        with patch('pandas.read_csv', side_effect=Exception("CSV parsing error")):
            with pytest.raises(ValueError) as excinfo:
                worker.run_graph_job(job_id, "s3://input-bucket/invalid.csv", mock_task_store)
            
            assert "Could not parse input file" in str(excinfo.value)
            assert "CSV parsing error" in str(excinfo.value)
            assert mock_task_store[job_id]["status"] == Status.FAILED
            assert "Could not parse input file" in mock_task_store[job_id]["error"]

# Test successful execution (end-to-end)
@patch('shutil.rmtree')  # Mock rmtree to avoid actual directory removal
def test_run_graph_job_success(mock_rmtree, mock_task_store, mock_s3_client, mock_s3_env, 
                              mock_graph, mock_notebook_exporter, temp_dir, job_id):
    # Patch the WORKER_TEMP_DIR to use our temp directory
    with patch('app.worker.WORKER_TEMP_DIR', Path(str(temp_dir))):
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
                    
        # Set up the mocks for S3 download/upload
        def mock_download_file(bucket, key, file_path):
            # Make sure destination directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            # Copy our test CSV to the destination
            shutil.copy(str(csv_file_path), file_path)
            
            mock_s3_client.download_file.side_effect = mock_download_file
            
            # Create a test file in output directory for upload testing
            output_file = output_dir.join("result.csv")
            output_file.write("result data")
            
            # Run the task
            result = worker.run_graph_job(job_id, "s3://input-bucket/test.csv", mock_task_store)
            
            # Verify the task executed correctly
            assert result is None  # run_graph_job doesn't return anything now
            
            # Check the task store was updated correctly
            assert mock_task_store[job_id]["status"] == Status.SUCCESS
            assert mock_task_store[job_id]["result"].startswith("s3://")
            assert "test-output-bucket" in mock_task_store[job_id]["result"]
            assert job_id in mock_task_store[job_id]["result"]
            
            # Verify graph was created and fit
            mock_graph.fit.assert_called_once_with(df=test_df)
            
            # Verify notebook exporter was called
            mock_notebook_exporter.export_notebook.assert_called_once()
            
            # Verify S3 uploads were performed
            assert mock_s3_client.upload_file.called
            
            # Verify cleanup was attempted
            assert mock_rmtree.called

#
# --- Tests for download_and_load_data ---

@pytest.fixture
def mock_pandas_read_csv():
    with patch('pandas.read_csv') as mock_read_csv:
        yield mock_read_csv

def test_download_and_load_data_s3_success(mock_s3_client, mock_pandas_read_csv, temp_dir, job_id):
    """Test successful download from S3 and loading data."""
    input_s3_uri = "s3://test-bucket/test.csv"
    local_target_dir = Path(str(temp_dir)) / "input" / job_id
    local_target_dir.mkdir(parents=True, exist_ok=True)
    local_target_file = local_target_dir / "test.csv"

    # Mock S3 download
    mock_s3_client.download_file.return_value = None
    
    # Mock _get_file_size to prevent FileNotFoundError if file isn't actually created by download_file mock
    with patch('app.worker._get_file_size', return_value=100):
        # Mock pd.read_csv
        mock_df = pd.DataFrame({'col1': [1], 'col2': [2]})
        mock_pandas_read_csv.return_value = mock_df

        # Patch WORKER_TEMP_DIR to control where local_input_file_path is constructed
        # The local_input_file_path in download_and_load_data is local_input_dir / input_uri.split('/')[-1]
        # So, we want local_input_file_path to be local_target_file
        # worker.WORKER_TEMP_DIR / "input" / job_id / "test.csv"
        
        # The local_input_file_path in the actual function is determined by its arguments,
        # so we just need to pass the correct one.
        # local_input_file_path in download_and_load_data is created as:
        # Path(os.path.join(WORKER_TEMP_DIR, "input", job_id, Path(input_uri).name))
        # We will use the local_target_file which we want it to resolve to.

        df = worker.download_and_load_data(job_id, input_s3_uri, local_target_file)

    mock_s3_client.download_file.assert_called_once_with("test-bucket", "test.csv", str(local_target_file))
    mock_pandas_read_csv.assert_called_once_with(local_target_file)
    pd.testing.assert_frame_equal(df, mock_df)

def test_download_and_load_data_local_success(temp_dir, job_id):
    """Test successful copy from local path and loading data."""
    
    input_df = generate_mock_data(n_units=100, n_periods=2, n_treated=50) # Smaller data for faster test
    
    # Create a temporary source CSV file
    source_dir = Path(str(temp_dir)) / "source_data_dir"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_file_path = source_dir / "source.csv"
    input_df.to_csv(source_file_path, index=False)
    
    # Define the target path where the function will copy the file
    # This path needs to be what download_and_load_data expects as its local_input_file_path argument.
    # Let's assume it's a specific structure within the temp_dir for the job
    target_copy_dir = Path(str(temp_dir)) / "worker_input_area" / job_id
    target_copy_dir.mkdir(parents=True, exist_ok=True)
    target_file_path = target_copy_dir / "data.csv" # The function will use this path to save and read

    # Call the function under test
    loaded_df = worker.download_and_load_data(job_id, str(source_file_path), target_file_path)

    # Assertions
    assert target_file_path.exists(), "File should have been copied to the target path"
    
    # Verify the content of the copied file (optional, but good for sanity)
    # copied_df_check = pd.read_csv(target_file_path)
    # pd.testing.assert_frame_equal(copied_df_check, input_df, check_dtype=False)

    # Verify the DataFrame returned by the function
    # When reading from CSV, dtypes might change (e.g., int to float if NaNs are present, though not with this mock data).
    # It's often good to compare with check_dtype=False or ensure dtypes are consistent after read_csv.
    pd.testing.assert_frame_equal(loaded_df, input_df, check_dtype=False)


def test_download_and_load_data_s3_non_csv_error(mock_s3_client, temp_dir, job_id):
    """Test error when S3 file is not a CSV."""
    input_s3_uri = "s3://test-bucket/test.txt"
    local_target_dir = Path(str(temp_dir)) / "input" / job_id
    local_target_dir.mkdir(parents=True, exist_ok=True)
    local_target_file = local_target_dir / "test.txt"

    mock_s3_client.download_file.return_value = None
    with patch('app.worker._get_file_size', return_value=100): # Mock file size
        # Create a dummy file locally to simulate download
        local_target_file.write_text("dummy content")

        with pytest.raises(ValueError) as excinfo:
            worker.download_and_load_data(job_id, input_s3_uri, local_target_file)
        
    assert "Input file must be a CSV file" in str(excinfo.value)

def test_download_and_load_data_local_non_csv_error(temp_dir, job_id):
    """Test error when local file is not a CSV."""
    source_file_path = temp_dir.join("source.txt")
    source_file_path.write("not a csv")

    local_target_dir = Path(str(temp_dir)) / "input" / job_id
    local_target_dir.mkdir(parents=True, exist_ok=True)
    local_target_file = local_target_dir / "source.txt"

    with patch('app.worker._get_file_size', return_value=100): # Mock file size
        with pytest.raises(ValueError) as excinfo:
            worker.download_and_load_data(job_id, str(source_file_path), local_target_file)

    assert "Input file must be a CSV file" in str(excinfo.value)


def test_download_and_load_data_s3_download_no_credentials_error(mock_s3_client, temp_dir, job_id):
    """Test NoCredentialsError during S3 download."""
    input_s3_uri = "s3://test-bucket/test.csv"
    local_target_file = Path(str(temp_dir)) / "input" / job_id / "test.csv"
    
    mock_s3_client.download_file.side_effect = NoCredentialsError()
    
    with pytest.raises(NoCredentialsError):
        worker.download_and_load_data(job_id, input_s3_uri, local_target_file)

def test_download_and_load_data_s3_download_client_error(mock_s3_client, temp_dir, job_id):
    """Test ClientError during S3 download."""
    input_s3_uri = "s3://test-bucket/test.csv"
    local_target_file = Path(str(temp_dir)) / "input" / job_id / "test.csv"

    client_error = ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "GetObject")
    mock_s3_client.download_file.side_effect = client_error
    
    with pytest.raises(FileNotFoundError): # We convert 404 to FileNotFoundError
        worker.download_and_load_data(job_id, input_s3_uri, local_target_file)

def test_download_and_load_data_local_file_not_found_error(temp_dir, job_id):
    """Test FileNotFoundError for local file copy."""
    non_existent_source_file = temp_dir.join("non_existent.csv")
    local_target_file = Path(str(temp_dir)) / "input" / job_id / "non_existent.csv"
    
    with pytest.raises(FileNotFoundError):
        worker.download_and_load_data(job_id, str(non_existent_source_file), local_target_file)


def test_download_and_load_data_csv_parsing_error(mock_s3_client, mock_pandas_read_csv, temp_dir, job_id):
    """Test error during CSV parsing."""
    input_s3_uri = "s3://test-bucket/bad.csv"
    local_target_dir = Path(str(temp_dir)) / "input" / job_id
    local_target_dir.mkdir(parents=True, exist_ok=True)
    local_target_file = local_target_dir / "bad.csv"

    mock_s3_client.download_file.return_value = None # Simulate successful download
    # Create a dummy file locally to simulate download
    local_target_file.write_text("bad,csv\ncontent")

    with patch('app.worker._get_file_size', return_value=100): # Mock file size
        mock_pandas_read_csv.side_effect = Exception("CSV parsing failed")
        with pytest.raises(ValueError) as excinfo:
            worker.download_and_load_data(job_id, input_s3_uri, local_target_file)

    assert "Could not parse input file" in str(excinfo.value)
    assert "CSV parsing failed" in str(excinfo.value)

# --- Tests for copy_s3 ---

def test_copy_s3_success(mock_s3_client, temp_dir, job_id):
    """Test successful S3 file download via copy_s3."""
    input_s3_uri = "s3://test-bucket/file.csv"
    local_target_path = Path(str(temp_dir)) / "local_file.csv"
    
    mock_s3_client.download_file.return_value = None
    with patch('app.worker._get_file_size', return_value=123): # Mock file size
        worker.copy_s3(job_id, input_s3_uri, local_target_path, time.time())
    
    mock_s3_client.download_file.assert_called_once_with("test-bucket", "file.csv", str(local_target_path))

def test_copy_s3_invalid_uri(job_id, temp_dir):
    """Test copy_s3 with an invalid S3 URI."""
    invalid_uri = "not-s3-uri/file.csv"
    local_target_path = Path(str(temp_dir)) / "local_file.csv"
    with pytest.raises(ValueError) as excinfo:
        worker.copy_s3(job_id, invalid_uri, local_target_path, time.time())
    assert "Invalid S3 URI format" in str(excinfo.value)

def test_copy_s3_no_credentials_error(mock_s3_client, temp_dir, job_id):
    """Test NoCredentialsError during S3 download in copy_s3."""
    input_s3_uri = "s3://test-bucket/file.csv"
    local_target_path = Path(str(temp_dir)) / "local_file.csv"
    
    mock_s3_client.download_file.side_effect = NoCredentialsError()
    
    with pytest.raises(NoCredentialsError):
        worker.copy_s3(job_id, input_s3_uri, local_target_path, time.time())

def test_copy_s3_client_error_404(mock_s3_client, temp_dir, job_id):
    """Test ClientError (404) during S3 download in copy_s3."""
    input_s3_uri = "s3://test-bucket/notfound.csv"
    local_target_path = Path(str(temp_dir)) / "local_file.csv"
    
    client_error = ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "GetObject")
    mock_s3_client.download_file.side_effect = client_error
    
    with pytest.raises(FileNotFoundError) as excinfo: # copy_s3 converts 404 to FileNotFoundError
        worker.copy_s3(job_id, input_s3_uri, local_target_path, time.time())
    assert f"The S3 object {input_s3_uri} does not exist" in str(excinfo.value)

def test_copy_s3_client_error_other(mock_s3_client, temp_dir, job_id):
    """Test other ClientError during S3 download in copy_s3."""
    input_s3_uri = "s3://test-bucket/some_error.csv"
    local_target_path = Path(str(temp_dir)) / "local_file.csv"
    
    client_error = ClientError({"Error": {"Code": "500", "Message": "Server Error"}}, "GetObject")
    mock_s3_client.download_file.side_effect = client_error
    
    with pytest.raises(ClientError) as excinfo:
        worker.copy_s3(job_id, input_s3_uri, local_target_path, time.time())
    assert "500" in str(excinfo.value) # Check if the original ClientError is raised

# --- Tests for copy_local ---

def test_copy_local_success(temp_dir, job_id):
    """Test successful local file copy via copy_local."""
    source_file = temp_dir.join("source_data.csv")
    source_file.write("data,content\n1,2")
    
    target_dir = Path(str(temp_dir)) / "target_dir"
    target_dir.mkdir(parents=True, exist_ok=True)
    local_target_path = target_dir / "copied_data.csv"
    
    with patch('app.worker._get_file_size', return_value=12): # Mock file size
        worker.copy_local(job_id, str(source_file), local_target_path, time.time())
    
    assert local_target_path.exists()
    assert local_target_path.read_text() == "data,content\n1,2"

def test_copy_local_source_not_found(temp_dir, job_id):
    """Test copy_local when source file does not exist."""
    non_existent_source = temp_dir.join("non_existent.csv")
    local_target_path = Path(str(temp_dir)) / "target_dir" / "file.csv"
    
    with pytest.raises(FileNotFoundError) as excinfo:
        worker.copy_local(job_id, str(non_existent_source), local_target_path, time.time())
    assert f"Local file not found: {str(non_existent_source)}" in str(excinfo.value)

def test_copy_local_destination_dir_not_found(temp_dir, job_id):
    """Test copy_local when destination directory does not exist."""
    source_file = temp_dir.join("source_data.csv")
    source_file.write("data,content\n1,2")
    
    # Target directory does not exist
    non_existent_target_dir = Path(str(temp_dir)) / "non_existent_target_dir"
    local_target_path = non_existent_target_dir / "copied_data.csv"
    
    with pytest.raises(FileNotFoundError) as excinfo: # shutil.copy2 raises FileNotFoundError if dst parent does not exist
        worker.copy_local(job_id, str(source_file), local_target_path, time.time())
    # The error message from shutil.copy2 might vary slightly, 
    # so checking for the path is a robust way
    assert str(local_target_path.parent) in str(excinfo.value) or "No such file or directory" in str(excinfo.value)

def test_copy_local_shutil_error(temp_dir, job_id):
    """Test copy_local when shutil.copy2 raises an unexpected error."""
    source_file = temp_dir.join("source_data.csv")
    source_file.write("data,content\n1,2")
    
    target_dir = Path(str(temp_dir)) / "target_dir"
    target_dir.mkdir(parents=True, exist_ok=True)
    local_target_path = target_dir / "copied_data.csv"
    
    with patch('shutil.copy2', side_effect=OSError("Disk full")):
        with pytest.raises(OSError) as excinfo:
            worker.copy_local(job_id, str(source_file), local_target_path, time.time())
        assert "Disk full" in str(excinfo.value)

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