import os
import boto3
from pathlib import Path
import logging
import pandas as pd
from botocore.exceptions import NoCredentialsError, ClientError
from typing import Union, Tuple, BinaryIO, Optional
import re
from enum import Enum

class Status(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"

# Configure logging
logger = logging.getLogger("file_io")

# Configure S3 client with timeouts
S3_REGION = os.getenv("AWS_REGION", "us-east-1")
s3_client = boto3.client(
    "s3", 
    region_name=S3_REGION,
    config=boto3.session.Config(
        connect_timeout=5,  # 5 seconds
        read_timeout=60,    # 60 seconds
        retries={'max_attempts': 3}
    )
)

def is_s3_path(path: str) -> bool:
    """
    Determine if a path is an S3 URI.
    
    Args:
        path: String path to check
        
    Returns:
        bool: True if the path is an S3 URI, False otherwise
    """
    return path.startswith("s3://")

def parse_path(path: str) -> Tuple[str, str, str]:
    """
    Parse a path string into components.
    
    For S3 URIs (s3://bucket/key):
        - scheme: "s3"
        - location: bucket name
        - path: key
        
    For local paths:
        - scheme: "file"
        - location: empty string
        - path: full local path
    
    Args:
        path: Path string to parse
        
    Returns:
        Tuple containing (scheme, location, path)
    """
    if is_s3_path(path):
        # Parse S3 URI
        path = path[5:]  # Remove "s3://"
        parts = path.split('/', 1)
        bucket = parts[0]
        # If there's a path component, return it, otherwise return empty string
        key = parts[1] if len(parts) > 1 else ""
        return "s3", bucket, key
    else:
        # Local path
        return "file", "", str(path)

def extract_filename(path: str) -> str:
    """
    Extract the filename from a path (local or S3).
    
    Args:
        path: Path string
        
    Returns:
        Filename portion of the path
    """
    if is_s3_path(path):
        _, _, key = parse_path(path)
        return os.path.basename(key)
    else:
        return os.path.basename(path)

def check_path_exists(path: str) -> bool:
    """
    Check if a path (S3 or local) exists.
    
    Args:
        path: Path to check
        
    Returns:
        bool: True if path exists, False otherwise
    """
    scheme, location, file_path = parse_path(path)
    
    if scheme == "s3":
        try:
            s3_client.head_object(Bucket=location, Key=file_path)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                return False
            # Re-raise other AWS errors
            raise
    else:
        # Local file check
        return os.path.exists(file_path)

def read_from(path: str) -> BinaryIO:
    """
    Read content from a path (S3 or local).
    
    Args:
        path: Path to read from
        
    Returns:
        File-like object with the content
    """
    import io
    import tempfile
    
    scheme, location, file_path = parse_path(path)
    
    if scheme == "s3":
        # For S3, we'll download to a temporary file or memory buffer
        # depending on expected size
        try:
            buffer = io.BytesIO()
            s3_client.download_fileobj(location, file_path, buffer)
            buffer.seek(0)
            return buffer

        except ClientError as e:
            logger.error(f"Error reading from S3: {e}")
            raise IOError(f"Failed to read from S3: {e}")
    else:
        # Local file
        try:
            return open(file_path, 'rb')
        except IOError as e:
            logger.error(f"Error reading local file: {e}")
            raise

def write_to(file_obj: BinaryIO, path: str) -> str:
    """
    Write content to a path (S3 or local).
    
    Args:
        file_obj: File-like object to read from
        path: Destination path
        
    Returns:
        The path where the file was written
    """
    scheme, location, file_path = parse_path(path)
    
    if scheme == "s3":
        try:
            # Reset file position to beginning if needed
            if hasattr(file_obj, 'seek'):
                file_obj.seek(0)
            
            # Upload to S3
            s3_client.upload_fileobj(file_obj, location, file_path)
            return f"s3://{location}/{file_path}"
        except (NoCredentialsError, ClientError) as e:
            logger.error(f"Error writing to S3: {e}")
            raise IOError(f"Failed to write to S3: {e}")
    else:
        # Local file
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write file
            with open(file_path, 'wb') as dest_file:
                if hasattr(file_obj, 'read'):
                    # Reset position if needed
                    if hasattr(file_obj, 'seek'):
                        file_obj.seek(0)
                    # Copy content
                    dest_file.write(file_obj.read())
                else:
                    # Handle string/bytes directly
                    dest_file.write(file_obj)
            return file_path
        except IOError as e:
            logger.error(f"Error writing local file: {e}")
            raise

def read_dataframe(path: str) -> pd.DataFrame:
    """
    Read a pandas DataFrame from a path (S3 or local).
    Automatically detects CSV, Excel, or other formats based on extension.
    
    Args:
        path: Path to read from
        
    Returns:
        pandas DataFrame
    """
    file_ext = os.path.splitext(path)[1].lower()
    
    with read_from(path) as file_obj:
        if file_ext == '.csv':
            return pd.read_csv(file_obj)
        elif file_ext in ['.xls', '.xlsx']:
            return pd.read_excel(file_obj)
        elif file_ext == '.json':
            return pd.read_json(file_obj)
        else:
            # Default to CSV
            return pd.read_csv(file_obj)

def write_dataframe(df: pd.DataFrame, path: str, index: bool = False) -> str:
    """
    Write a pandas DataFrame to a path (S3 or local).
    Format determined by file extension.
    
    Args:
        df: DataFrame to write
        path: Destination path
        index: Whether to include index in output
        
    Returns:
        The path where the file was written
    """
    import io
    
    file_ext = os.path.splitext(path)[1].lower()
    buffer = io.BytesIO()
    
    if file_ext == '.csv':
        df.to_csv(buffer, index=index)
    elif file_ext in ['.xls', '.xlsx']:
        df.to_excel(buffer, index=index)
    elif file_ext == '.json':
        df.to_json(buffer)
    else:
        # Default to CSV
        df.to_csv(buffer, index=index)
    
    return write_to(buffer, path)

def join_paths(base_path: str, *parts: str) -> str:
    """
    Join path components, maintaining the scheme (s3:// or local).
    
    Args:
        base_path: Base path (can be S3 URI or local path)
        *parts: Additional path components to join
        
    Returns:
        Joined path string
    """
    if is_s3_path(base_path):
        scheme, bucket, key = parse_path(base_path)
        
        # Start with the bucket
        result = f"s3://{bucket}"
        
        # Join key with parts, handling empty keys
        all_parts = [key] if key else []
        all_parts.extend(parts)
        
        # Remove empty parts and normalize slashes
        all_parts = [p.strip('/') for p in all_parts if p]
        
        if all_parts:
            result += "/" + "/".join(all_parts)
            
        return result
    else:
        # Use os.path.join for local paths
        return os.path.join(base_path, *parts)
    
    # Function to parse S3 URI into bucket and path components
def parse_s3_uri(s3_uri):
    """Parse an S3 URI into bucket and key components."""
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