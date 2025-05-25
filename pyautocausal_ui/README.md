# PyAutoCausal UI

A lightweight HTML + JavaScript front-end for interacting with the PyAutoCausal backend service.

## Features

1. Upload a CSV dataset directly to an S3 bucket shared with the backend.
2. Submit a job to the FastAPI `/jobs` endpoint with the uploaded S3 path.
3. Poll for job completion every 10 seconds.
4. Download the resulting analysis artifact from S3 when the job completes.

## Quick Start (Local Development)

1. **Set AWS credentials & config**  
   Edit `config.js` and provide:

   ```js
   window.PYAUTOCAUSAL_UI_CONFIG = {
     apiBaseUrl: "http://localhost:8000",  // FastAPI base URL
     awsRegion: "us-east-1",               // AWS region
     s3Bucket: "pyautocausal",              // Bucket name
     s3InputPrefix: "user_inputs",          // Folder for uploads

     // ⚠️ Development credentials only!
     awsAccessKeyId: "YOUR_ACCESS_KEY_ID",
     awsSecretAccessKey: "YOUR_SECRET_ACCESS_KEY"
   };
   ```

   **Important:** For production you should _not_ embed static credentials in the UI. Instead use pre-signed URLs or an identity provider such as Amazon Cognito.

2. **Serve the files**  
   Any simple HTTP server will work, e.g.:

   ```bash
   # From repository root
   cd pyautocausal_ui
   python -m http.server 9000
   ```

   Then open <http://localhost:9000> in your browser.

3. **Run the backend**  
   Ensure the FastAPI service is running locally (default <http://localhost:8000>). Then you can test the full flow:

   1. Select a CSV file
   2. Click "Upload & Run Analysis"
   3. Wait until the job completes and click "Download Results"

## Next Steps / Improvements

- Replace static credentials with secure auth (Cognito, pre-signed uploads)
- Display progress bars / graphs for job status
- Add drag-and-drop support for file uploads
- Bundle the UI with a modern framework (React/Vite) if you outgrow the current setup 