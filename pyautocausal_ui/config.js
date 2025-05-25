// pyautocausal_ui configuration
// IMPORTANT: Do NOT commit real AWS credentials. For local development only.
// In production, use a secure method (e.g., Cognito, STS, pre-signed URLs) to handle uploads.

window.PYAUTOCAUSAL_UI_CONFIG = {
  // Base URL where the FastAPI backend is running
  apiBaseUrl: "http://ec2-3-141-193-89.us-east-2.compute.amazonaws.com:8000",  // Updated to point to remote FastAPI instance on port 8000

  // AWS / S3 configuration
  awsRegion: "us-east-2",
  s3Bucket: "pyautocausal",
  s3InputPrefix: "user_inputs"
}; 