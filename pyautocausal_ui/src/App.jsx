import React, { useState, useEffect, useRef } from 'react';
import Papa from 'papaparse';
import usePipelines from './hooks/usePipelines';
import PipelineSelect from './components/PipelineSelect';
import FileInput from './components/FileInput';
import ColumnMapping from './components/ColumnMapping';

export default function App() {
  const { pipelines, loading, error } = usePipelines();
  const [selectedPipeline, setSelectedPipeline] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [csvHeaders, setCsvHeaders] = useState([]);
  const [columnMapping, setColumnMapping] = useState({});
  const [jobStatus, setJobStatus] = useState('');
  const [downloadUrl, setDownloadUrl] = useState(null);
  const pollIntervalRef = useRef(null);

  const stopPolling = () => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  };

  const handleFileSelect = (file) => {
    stopPolling();
    setSelectedFile(file);
    setColumnMapping({}); // Reset mapping when file changes
    setJobStatus(''); // Reset status
    setDownloadUrl(null);
    if (file) {
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        preview: 1, // Only read the first row for headers
        complete: (results) => {
          setCsvHeaders(results.meta.fields);
        },
      });
    } else {
      setCsvHeaders([]);
    }
  };

  const handlePipelineChange = (pipelineName) => {
    stopPolling();
    setSelectedPipeline(pipelineName);
    setColumnMapping({}); // Also reset mapping
    setJobStatus('');
    setDownloadUrl(null);
  };

  const handleMappingChange = (pipelineColumn, csvColumn) => {
    setColumnMapping((prev) => ({
      ...prev,
      [pipelineColumn]: csvColumn,
    }));
  };
  
  const getSelectedPipelineObject = () => {
    if (!selectedPipeline) return null;
    return pipelines.find(p => p.name === selectedPipeline);
  }

  const uploadFileToS3 = async (file) => {
    const { awsRegion, s3Bucket, s3InputPrefix } = window.PYAUTOCAUSAL_UI_CONFIG;
    if (!awsRegion || !s3Bucket || !s3InputPrefix) {
        throw new Error('S3 configuration is missing from window.PYAUTOCAUSAL_UI_CONFIG');
    }
    const key = `${s3InputPrefix}/${Date.now()}_${file.name}`;
    const uploadUrl = `https://${s3Bucket}.s3.${awsRegion}.amazonaws.com/${key}`;

    const res = await fetch(uploadUrl, {
      method: 'PUT',
      headers: { 'Content-Type': file.type || 'text/csv' },
      body: file,
    });

    if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`S3 Upload failed: ${res.status} ${errorText}`);
    }
    return `s3://${s3Bucket}/${key}`;
  };

  const pollJobStatus = async (statusUrl) => {
    try {
      const res = await fetch(statusUrl);
      if (!res.ok) {
        // Stop polling on persistent errors, but not transient ones
        if (res.status === 404 || res.status === 403) {
          stopPolling();
          setJobStatus(`Error: Job status endpoint not found. Polling stopped.`);
        }
        return; // Otherwise, keep trying
      }

      const data = await res.json();
      console.log("Polling response:", data); // Add console log for debugging

      // Always update the status message from the backend
      const message = `Status: ${data.status} – ${data.message}`;
      setJobStatus(message);

      if (data.status === 'COMPLETED' || data.status === 'FAILED') {
        stopPolling();
        if (data.status === 'COMPLETED' && data.download_url) {
            setDownloadUrl(data.download_url); // Set the download URL
        }
      }
    } catch (err) {
      console.error('Polling error:', err);
      stopPolling();
      setJobStatus('Error fetching job status due to network issue. Polling stopped.');
    }
  };

  const handleSubmit = async () => {
    if (!isSubmittable || !selectedFile) return;

    stopPolling();
    setJobStatus('Uploading file...');
    setDownloadUrl(null);
    try {
      const s3Uri = await uploadFileToS3(selectedFile);
      setJobStatus('File uploaded. Submitting job...');

      const jobPayload = {
        input_path: s3Uri,
        pipeline_name: selectedPipeline,
        column_mapping: columnMapping,
      };

      const res = await fetch(`${window.PYAUTOCAUSAL_UI_CONFIG.apiBaseUrl}/jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(jobPayload),
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Job submission failed: ${res.status} – ${errText}`);
      }

      const jobResult = await res.json();
      setJobStatus('Job submitted! Monitoring status...');
      
      let statusUrl = jobResult.status_url;
      // If we're hitting the production API, ensure we poll over HTTPS,
      // as Caddy will redirect HTTP->HTTPS and cause issues with fetch.
      const url = new URL(statusUrl);
      if (url.hostname === 'api.pyautocausal.com' && url.protocol === 'http:') {
        url.protocol = 'https';
        statusUrl = url.toString();
      }
      
      // Start polling
      pollIntervalRef.current = setInterval(() => {
        pollJobStatus(statusUrl);
      }, 3000); // Poll every 3 seconds

    } catch (err) {
      console.error(err);
      setJobStatus(`Error: ${err.message}`);
    }
  };

  // Cleanup effect to stop polling when the component unmounts
  useEffect(() => {
    return () => stopPolling();
  }, []);

  const showMapping = selectedPipeline && selectedFile && csvHeaders.length > 0;
  const pipelineObject = getSelectedPipelineObject();
  
  const requiredColumns = pipelineObject?.required_columns || [];
  const allRequiredColumnsMapped = requiredColumns.every(
    (pColId) => columnMapping[pColId]
  );
  const isSubmittable = showMapping && allRequiredColumnsMapped;

  return (
    <main style={{ padding: '1rem' }}>
      <h1>PyAutoCausal React UI</h1>
      <div style={{ display: 'flex', gap: '2rem', marginBottom: '1rem' }}>
        <PipelineSelect
          pipelines={pipelines}
          loading={loading}
          error={error}
          value={selectedPipeline}
          onChange={handlePipelineChange}
        />
        <FileInput onFileSelect={handleFileSelect} />
      </div>
      
      {showMapping && pipelineObject && (
        <>
          <ColumnMapping
            csvHeaders={csvHeaders}
            pipeline={pipelineObject}
            onMappingChange={handleMappingChange}
          />
          <button
            onClick={handleSubmit}
            disabled={!isSubmittable}
            style={{ marginTop: '1rem' }}
          >
            Submit Job
          </button>
          {jobStatus && <p style={{ marginTop: '1rem' }}>{jobStatus}</p>}
          {downloadUrl && (
            <a
              href={downloadUrl}
              download
              className="button"
              style={{ marginTop: '1rem', display: 'inline-block' }}
            >
              Download Results
            </a>
          )}
        </>
      )}
    </main>
  );
} 