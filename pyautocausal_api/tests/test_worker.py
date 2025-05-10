import os
import pytest
import uuid
import pandas as pd
from pathlib import Path
import shutil
from unittest.mock import patch, MagicMock, Mock
from botocore.exceptions import NoCredentialsError, ClientError

# Import the Celery app and worker functions
from app import worker
from app.worker import parse_s3_uri

# Access the underlying run function wrapped by Celery
run_graph_job_func = worker.run_graph_job.__wrapped__

# Fixture to provide a UUID for testing
@pytest.fixture
def job_id():
    return str(uuid.uuid4())

# Fixture to mock task self object
@pytest.fixture
def mock_task_self():
    mock_self = MagicMock()
    mock_self.update_state = MagicMock()
    
    # The key issue: mock_self.request.id must be properly set for update_state to work
    mock_self.request = MagicMock()
    mock_self.request.id = "test-task-id"
    
    # Better mocking of the backend to handle update_state properly
    mock_backend = MagicMock()
    mock_backend.store_result = MagicMock()
    mock_backend.get_key_for_task = MagicMock(return_value="test-key")
    mock_backend.get = MagicMock(return_value={"status": "PENDING", "result": None})
    
    mock_self.backend = mock_backend
    return mock_self

# Fixture to mock the S3 client
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
        with patch('app.worker.S3_OUTPUT_BUCKET', 's3://test-output-bucket/output/path'):
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

# Test run_graph_job missing S3_OUTPUT_BUCKET
def test_run_graph_job_missing_output_bucket(mock_task_self):
    with patch('app.worker.S3_OUTPUT_BUCKET', None):
        with pytest.raises(ValueError) as excinfo:
            bound_func = run_graph_job_func.__get__(mock_task_self, type(mock_task_self))
            bound_func("job-id", "s3://input-bucket/file.csv", "file.csv")
        
        assert "S3 output bucket not specified" in str(excinfo.value)

# Test run_graph_job with invalid S3_OUTPUT_BUCKET format
def test_run_graph_job_invalid_output_bucket_format(mock_task_self):
    with patch('app.worker.S3_OUTPUT_BUCKET', 'invalid-format'):
        with pytest.raises(ValueError) as excinfo:
            bound_func = run_graph_job_func.__get__(mock_task_self, type(mock_task_self))
            bound_func("job-id", "s3://input-bucket/file.csv", "file.csv")
        
        assert "Invalid S3 output bucket URI format" in str(excinfo.value)

# Test S3 download failure (NoCredentialsError)
def test_run_graph_job_s3_download_credentials_error(mock_task_self, mock_s3_client, mock_s3_env):
    # Make S3 download fail with NoCredentialsError
    mock_s3_client.download_file.side_effect = NoCredentialsError()
    
    with pytest.raises(NoCredentialsError):
        bound_func = run_graph_job_func.__get__(mock_task_self, type(mock_task_self))
        bound_func("job-id", "s3://input-bucket/file.csv", "file.csv")
    
    # In testing mode, update_state is not called
    # So no need to verify update_state calls

# Test S3 download failure (ClientError)
def test_run_graph_job_s3_download_client_error(mock_task_self, mock_s3_client, mock_s3_env):
    # Make S3 download fail with ClientError
    mock_s3_client.download_file.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."}},
        "GetObject"
    )
    
    with pytest.raises(ClientError):
        bound_func = run_graph_job_func.__get__(mock_task_self, type(mock_task_self))
        bound_func("job-id", "s3://input-bucket/file.csv", "file.csv")

# Test CSV parsing error
def test_run_graph_job_csv_parsing_error(mock_task_self, mock_s3_client, mock_s3_env, temp_dir):
    # Patch the WORKER_TEMP_DIR to use our temp directory
    with patch('app.worker.WORKER_TEMP_DIR', Path(str(temp_dir))):
        # Create an invalid CSV file
        temp_file = temp_dir.join("job_id/input/invalid.csv")
        temp_file.write("this,is,not,a,valid\ncsv,file\nwith,inconsistent,columns")
        
        # Mock successful S3 download
        def mock_download_file(bucket, key, file_path):
            # Instead of actually downloading, we just ensure the target file exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(temp_file), file_path)
        
        mock_s3_client.download_file.side_effect = mock_download_file
        
        # Mock pd.read_csv to raise an exception
        with patch('pandas.read_csv', side_effect=Exception("CSV parsing error")):
            with pytest.raises(ValueError) as excinfo:
                bound_func = run_graph_job_func.__get__(mock_task_self, type(mock_task_self))
                bound_func("job-id", "s3://input-bucket/invalid.csv", "invalid.csv")
            
            assert "Could not parse input file" in str(excinfo.value)
            assert "CSV parsing error" in str(excinfo.value)

# Test successful execution (end-to-end)
@patch('shutil.rmtree')  # Mock rmtree to avoid actual directory removal
def test_run_graph_job_success(mock_rmtree, mock_task_self, mock_s3_client, mock_s3_env, 
                              mock_graph, mock_notebook_exporter, temp_dir):
    # Patch the WORKER_TEMP_DIR to use our temp directory
    with patch('app.worker.WORKER_TEMP_DIR', Path(str(temp_dir))):
        # Create test directories and a sample CSV
        input_dir = temp_dir.join("job_id/input")
        output_dir = temp_dir.join("job_id/output")
        input_dir.ensure_dir()
        output_dir.ensure_dir()
        
        # Create a test CSV file
        csv_content = "col1,col2,col3\n1,2,3\n4,5,6"
        csv_file = input_dir.join("test.csv")
        csv_file.write(csv_content)
        
        # Patch pd.read_csv to return a DataFrame
        test_df = pd.DataFrame({'col1': [1, 4], 'col2': [2, 5], 'col3': [3, 6]})
        with patch('pandas.read_csv', return_value=test_df):
            
            # Set up the mocks for S3 download/upload
            def mock_download_file(bucket, key, file_path):
                # Make sure destination directory exists
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                # Copy our test CSV to the destination
                shutil.copy(str(csv_file), file_path)
            
            mock_s3_client.download_file.side_effect = mock_download_file
            
            # Create a test file in output directory for upload testing
            output_file = output_dir.join("result.csv")
            output_file.write("result data")
            
            # Run the task
            job_id = "job-id"
            input_s3_uri = "s3://input-bucket/test.csv"
            original_filename = "test.csv"
            
            result = run_graph_job_func.__get__(mock_task_self, type(mock_task_self))(job_id, input_s3_uri, original_filename)
            
            # Verify the task executed correctly
            assert result.startswith("s3://")
            assert "test-output-bucket" in result
            assert job_id in result
            
            # In testing mode, update_state is not called
            # So no need to verify update_state calls
            
            # Verify graph was created and fit
            mock_graph.fit.assert_called_once_with(df=test_df)
            
            # Verify notebook exporter was called
            mock_notebook_exporter.export_notebook.assert_called_once()
            
            # Verify S3 uploads were performed
            assert mock_s3_client.upload_file.called
            
            # Verify cleanup was attempted
            assert mock_rmtree.called

# Test S3 upload failure
def test_run_graph_job_s3_upload_failure(mock_task_self, mock_s3_client, mock_s3_env, 
                                         mock_graph, mock_notebook_exporter, temp_dir):
    # Patch the WORKER_TEMP_DIR to use our temp directory
    with patch('app.worker.WORKER_TEMP_DIR', Path(str(temp_dir))):
        # Create test directories and a sample CSV
        input_dir = temp_dir.join("job_id/input")
        output_dir = temp_dir.join("job_id/output")
        input_dir.ensure_dir()
        output_dir.ensure_dir()
        
        # Create a test CSV file
        csv_content = "col1,col2,col3\n1,2,3\n4,5,6"
        csv_file = input_dir.join("test.csv")
        csv_file.write(csv_content)
        
        # Mock successful S3 download
        def mock_download_file(bucket, key, file_path):
            # Make sure destination directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            # Copy our test CSV to the destination
            shutil.copy(str(csv_file), file_path)
        
        mock_s3_client.download_file.side_effect = mock_download_file
        
        # Make S3 upload fail with NoCredentialsError
        mock_s3_client.upload_file.side_effect = NoCredentialsError()
        
        # Patch pd.read_csv to return a DataFrame
        test_df = pd.DataFrame({'col1': [1, 4], 'col2': [2, 5], 'col3': [3, 6]})
        with patch('pandas.read_csv', return_value=test_df):
            # Create a test file in output directory for upload testing
            output_file = output_dir.join("result.csv")
            output_file.write("result data")
            
            # Run the task
            with pytest.raises(NoCredentialsError):
                bound_func = run_graph_job_func.__get__(mock_task_self, type(mock_task_self))
                bound_func("job-id", "s3://input-bucket/test.csv", "test.csv")
            
            # Verify graph was created and fit (meaning the task got to that point)
            mock_graph.fit.assert_called_once_with(df=test_df)

# Test notebook export failure (non-critical)
def test_run_graph_job_notebook_export_failure(mock_task_self, mock_s3_client, mock_s3_env, 
                                              mock_graph, mock_notebook_exporter, temp_dir):
    # Patch the WORKER_TEMP_DIR to use our temp directory
    with patch('app.worker.WORKER_TEMP_DIR', Path(str(temp_dir))):
        # Create test directories and a sample CSV
        input_dir = temp_dir.join("job_id/input")
        output_dir = temp_dir.join("job_id/output")
        input_dir.ensure_dir()
        output_dir.ensure_dir()
        
        # Create a test CSV file
        csv_content = "col1,col2,col3\n1,2,3\n4,5,6"
        csv_file = input_dir.join("test.csv")
        csv_file.write(csv_content)
        
        # Mock successful S3 download
        def mock_download_file(bucket, key, file_path):
            # Make sure destination directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            # Copy our test CSV to the destination
            shutil.copy(str(csv_file), file_path)
        
        mock_s3_client.download_file.side_effect = mock_download_file
        
        # Make notebook export fail
        mock_notebook_exporter.export_notebook.side_effect = Exception("Notebook export failed")
        
        # Patch pd.read_csv to return a DataFrame
        test_df = pd.DataFrame({'col1': [1, 4], 'col2': [2, 5], 'col3': [3, 6]})
        with patch('pandas.read_csv', return_value=test_df):
            # Create a test file in output directory for upload testing
            output_file = output_dir.join("result.csv")
            output_file.write("result data")
            
            # Run the task - this should still succeed despite notebook export failure
            result = run_graph_job_func.__get__(mock_task_self, type(mock_task_self))("job-id", "s3://input-bucket/test.csv", "test.csv")
            
            # Verify the task executed correctly
            assert result.startswith("s3://")
            assert "test-output-bucket" in result
            
            # Verify notebook export was attempted
            mock_notebook_exporter.export_notebook.assert_called_once()
            
            # Verify S3 uploads were still performed
            assert mock_s3_client.upload_file.called 