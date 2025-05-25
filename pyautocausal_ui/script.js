/**
 * PyAutoCausal UI Front-end Logic
 * ------------------------------------------------
 * Handles:
 * 1. Uploading a CSV file to the configured S3 bucket.
 * 2. Submitting a job to the FastAPI backend with the S3 path.
 * 3. Polling job status until completion or failure.
 * 4. Downloading result artifact(s) from S3 once the job is done.
 * ------------------------------------------------
 * NOTE: This implementation assumes you are working in a development
 * environment where your browser has permission to upload to S3 using
 * static credentials. In production, you should replace this approach
 * with pre-signed URLs or an identity provider (e.g., Cognito).
 */

(() => {
  // DOM elements
  const fileInput = document.getElementById("file-input");
  const uploadBtn = document.getElementById("upload-btn");
  const progressBar = document.getElementById("upload-progress");
  const statusDiv = document.getElementById("status");
  const downloadBtn = document.getElementById("download-btn");

  // Runtime state
  let jobStatusUrl = null;
  let resultS3Path = null;
  let pollIntervalId = null;
  const POLL_INTERVAL_MS = 10000; // 10s

  // Utility helpers -------------------------------------------------------
  const setStatus = (message) => {
    statusDiv.textContent = message;
  };

  // Upload the selected file to S3 and return the resulting s3:// URI.
  const uploadFileToS3 = async (file) => {
    const { awsRegion, s3Bucket, s3InputPrefix } = window.PYAUTOCAUSAL_UI_CONFIG;
    const key = `${s3InputPrefix}/${Date.now()}_${file.name}`;
    const uploadUrl = `https://${s3Bucket}.s3.${awsRegion}.amazonaws.com/${key}`;

    const res = await fetch(uploadUrl, {
      method: "PUT",
      headers: { "Content-Type": file.type || "text/csv" },
      body: file,
    });

    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    return `s3://${s3Bucket}/${key}`;
  };

  // Submit the job to the FastAPI backend --------------------------------
  const submitJob = async (inputPath) => {
    const formData = new FormData();
    formData.append("input_path", inputPath);

    // Log the request details for debugging
    console.log("Submitting job to:", `${window.PYAUTOCAUSAL_UI_CONFIG.apiBaseUrl}/jobs`);
    for (let [key, value] of formData.entries()) {
      console.log(`FormData: ${key} = ${value}`);
    }

    const res = await fetch(`${window.PYAUTOCAUSAL_UI_CONFIG.apiBaseUrl}/jobs`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Job submission failed: ${res.status} – ${errText}`);
    }

    return res.json();
  };

  // Poll status endpoint until job completes or fails --------------------
  const pollJobStatus = async () => {
    try {
      const res = await fetch(jobStatusUrl);
      if (!res.ok) throw new Error("Failed to fetch job status.");

      const data = await res.json();
      setStatus(`Job Status: ${data.status} – ${data.message}`);

      if (data.status === "COMPLETED") {
        clearInterval(pollIntervalId);
        resultS3Path = data.result_path;
        downloadBtn.style.display = "inline-block";
        setStatus("Job completed! You can now download the results.");
      } else if (data.status === "FAILED") {
        clearInterval(pollIntervalId);
        setStatus(`Job failed: ${data.error_details || "Unknown error"}`);
      }
    } catch (err) {
      console.error(err);
      // Keep polling unless a permanent failure occurred.
    }
  };

  // Download the generated results from S3 --------------------------------
  const downloadResults = async () => {
    if (!resultS3Path) return;

    const { awsRegion } = window.PYAUTOCAUSAL_UI_CONFIG;
    const [ , bucket, key ] = resultS3Path.match(/^s3:\/\/([^/]+)\/(.+)$/);
    const url = `https://${bucket}.s3.${awsRegion}.amazonaws.com/${key}`;

    const res = await fetch(url);
    if (!res.ok) { setStatus("Download failed"); return; }

    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = key.split("/").pop();
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  // Event bindings --------------------------------------------------------
  uploadBtn.addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file) {
      alert("Please select a CSV file first.");
      return;
    }

    try {
      setStatus("Uploading file to S3…");
      const s3Uri = await uploadFileToS3(file);
      setStatus("Upload complete. Submitting job…");

      const jobResp = await submitJob(s3Uri);
      jobStatusUrl = jobResp.status_url;
      setStatus("Job submitted! Awaiting results…");

      pollIntervalId = setInterval(pollJobStatus, POLL_INTERVAL_MS);
    } catch (err) {
      console.error(err);
      setStatus(`Error: ${err.message}`);
    }
  });

  downloadBtn.addEventListener("click", downloadResults);
})(); 