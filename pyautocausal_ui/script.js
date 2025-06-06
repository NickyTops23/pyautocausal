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
  const statusDiv = document.getElementById("status");
  const downloadBtn = document.getElementById("download-btn");
  const pipelineSelect = document.getElementById("pipeline-select");
  const pipelineDescription = document.getElementById("pipeline-description");
  const mappingSection = document.getElementById("mapping-section");
  const mappingTable = document.getElementById("mapping-table");
  const optionalMappingTable = document.getElementById("optional-mapping-table");

  // Runtime state ---------------------------------------------------------
  const state = {
    pipelines: [],
    selectedPipeline: null,   // full metadata object
    csvHeaders: [],
    jobStatusUrl: null,
    resultS3Path: null,
    pollIntervalId: null,
  };
  const POLL_INTERVAL_MS = 10000; // 10s

  // Utility helpers -------------------------------------------------------
  const setStatus = (message) => {
    statusDiv.textContent = message;
  };

  //-----------------------------------------------------------------------
  // 1.  Fetch available pipelines on load
  //-----------------------------------------------------------------------
  const fetchPipelines = async () => {
    try {
      const res = await fetch(`${window.PYAUTOCAUSAL_UI_CONFIG.apiBaseUrl}/pipelines`);
      if (!res.ok) throw new Error(`Failed to fetch pipelines: ${res.status}`);
      const data = await res.json();
      state.pipelines = data;

      // Populate <select>
      pipelineSelect.innerHTML = '<option value="">-- Select a pipeline --</option>' +
        data.map((p) => `<option value="${p.name}">${p.display_name || p.name}</option>`).join("");
      pipelineSelect.disabled = false;
    } catch (err) {
      console.error(err);
      setStatus(`Error loading pipelines: ${err.message}`);
    }
  };

  //-----------------------------------------------------------------------
  // 2.  CSV header parsing helper
  //-----------------------------------------------------------------------
  const parseCsvHeaders = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const firstLine = reader.result.split(/\r?\n/)[0];
        resolve(firstLine.split(',').map((h) => h.trim()));
      };
      reader.onerror = reject;
      reader.readAsText(file.slice(0, 4096)); // only need first chunk
    });
  };

  //-----------------------------------------------------------------------
  // 3.  Mapping UI builder + validation
  //-----------------------------------------------------------------------
  const buildMappingUI = () => {
    // Guard – need both pipeline & headers
    if (!state.selectedPipeline || state.csvHeaders.length === 0) {
      mappingSection.style.display = 'none';
      uploadBtn.disabled = true;
      return;
    }

    mappingSection.style.display = 'block';
    pipelineDescription.textContent = state.selectedPipeline.description || '';

    // Clear existing rows
    mappingTable.innerHTML = '';
    optionalMappingTable.innerHTML = '';

    // Helper to build a single row
    const addRow = (tableEl, col, isRequired) => {
      const tr = document.createElement('tr');
      tr.className = `mapping-row ${isRequired ? 'required' : 'optional'}`;
      tr.dataset.colId = col.id;

      const labelTd = document.createElement('td');
      labelTd.textContent = col.label || col.id;

      const selectTd = document.createElement('td');
      const select = document.createElement('select');

      // First option
      const placeholderOpt = document.createElement('option');
      placeholderOpt.value = '';
      placeholderOpt.textContent = isRequired ? '-- Select column --' : '-- (none) --';
      select.appendChild(placeholderOpt);

      // Add csv headers
      state.csvHeaders.forEach((hdr) => {
        const opt = document.createElement('option');
        opt.value = hdr;
        opt.textContent = hdr;
        select.appendChild(opt);
      });

      select.addEventListener('change', validateReadyState);
      selectTd.appendChild(select);

      tr.appendChild(labelTd);
      tr.appendChild(selectTd);
      tableEl.appendChild(tr);
    };

    // Required columns
    (state.selectedPipeline.required_columns || []).forEach((col) => addRow(mappingTable, col, true));
    // Optional columns
    (state.selectedPipeline.optional_columns || []).forEach((col) => addRow(optionalMappingTable, col, false));

    validateReadyState();
  };

  const validateReadyState = () => {
    const fileChosen = !!fileInput.files[0];
    const pipelineChosen = !!state.selectedPipeline;
    // ensure all required selects have a value
    const requiredSelects = mappingTable.querySelectorAll('tr.required select');
    const allRequiredMapped = Array.from(requiredSelects).every((sel) => sel.value);

    uploadBtn.disabled = !(fileChosen && pipelineChosen && allRequiredMapped);
  };

  //-----------------------------------------------------------------------
  // 4.  S3 upload helper
  //-----------------------------------------------------------------------
  const uploadFileToS3 = async (file) => {
    const { awsRegion, s3Bucket, s3InputPrefix } = window.PYAUTOCAUSAL_UI_CONFIG;
    const key = `${s3InputPrefix}/${Date.now()}_${file.name}`;
    const uploadUrl = `https://${s3Bucket}.s3.${awsRegion}.amazonaws.com/${key}`;

    const res = await fetch(uploadUrl, {
      method: 'PUT',
      headers: { 'Content-Type': file.type || 'text/csv' },
      body: file,
    });

    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    return `s3://${s3Bucket}/${key}`;
  };

  //-----------------------------------------------------------------------
  // 5.  Job submission + polling
  //-----------------------------------------------------------------------
  const submitJob = async (inputPath, pipelineName, columnMapping) => {
    const body = JSON.stringify({
      input_path: inputPath,
      pipeline_name: pipelineName,
      column_mapping: columnMapping,
    });

    const res = await fetch(`${window.PYAUTOCAUSAL_UI_CONFIG.apiBaseUrl}/jobs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Job submission failed: ${res.status} – ${errText}`);
    }

    return res.json(); // expecting { status_url: "…" }
  };

  const pollJobStatus = async () => {
    try {
      const res = await fetch(state.jobStatusUrl);
      if (!res.ok) throw new Error('Failed to fetch job status.');

      const data = await res.json();
      setStatus(`Job Status: ${data.status} – ${data.message}`);

      if (data.status === 'COMPLETED') {
        clearInterval(state.pollIntervalId);
        state.resultS3Path = data.result_path;
        downloadBtn.style.display = 'inline-block';
        setStatus('Job completed! You can now download the results.');
      } else if (data.status === 'FAILED') {
        clearInterval(state.pollIntervalId);
        setStatus(`Job failed: ${data.error_details || 'Unknown error'}`);
      }
    } catch (err) {
      console.error(err);
      // Keep polling unless a permanent failure occurred.
    }
  };

  const downloadResults = async () => {
    if (!state.resultS3Path) return;

    const { awsRegion } = window.PYAUTOCAUSAL_UI_CONFIG;
    const [, bucket, key] = state.resultS3Path.match(/^s3:\/\/([^/]+)\/(.+)$/);
    const url = `https://${bucket}.s3.${awsRegion}.amazonaws.com/${key}`;

    const res = await fetch(url);
    if (!res.ok) {
      setStatus('Download failed');
      return;
    }

    const blob = await res.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = key.split('/').pop();
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  //-----------------------------------------------------------------------
  // 6.  Event bindings
  //-----------------------------------------------------------------------
  pipelineSelect.addEventListener('change', () => {
    state.selectedPipeline = state.pipelines.find((p) => p.name === pipelineSelect.value) || null;
    buildMappingUI();
  });

  fileInput.addEventListener('change', async () => {
    const file = fileInput.files[0];
    if (!file) return;

    try {
      state.csvHeaders = await parseCsvHeaders(file);
      buildMappingUI();
    } catch (err) {
      console.error(err);
      setStatus(`Failed to read CSV headers: ${err.message}`);
    }
  });

  uploadBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file || !state.selectedPipeline) return;

    // Build mapping object from UI
    const mapping = {};
    document.querySelectorAll('tr.mapping-row').forEach((row) => {
      const colId = row.dataset.colId;
      const sel = row.querySelector('select');
      if (sel.value) mapping[colId] = sel.value;
    });

    try {
      setStatus('Uploading file to S3…');
      const s3Uri = await uploadFileToS3(file);
      setStatus('Upload complete. Submitting job…');

      const jobResp = await submitJob(s3Uri, state.selectedPipeline.name, mapping);
      state.jobStatusUrl = jobResp.status_url;
      setStatus('Job submitted! Awaiting results…');

      state.pollIntervalId = setInterval(pollJobStatus, POLL_INTERVAL_MS);
    } catch (err) {
      console.error(err);
      setStatus(`Error: ${err.message}`);
    }
  });

  downloadBtn.addEventListener('click', downloadResults);

  // Kick‐off initial data fetch
  fetchPipelines();
})(); 