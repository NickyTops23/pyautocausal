import React, { useState, useEffect } from 'react';
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

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setColumnMapping({}); // Reset mapping when file changes
    setJobStatus(''); // Reset status
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

  const handleSubmit = async () => {
    if (!isSubmittable || !selectedFile) return;

    setJobStatus('Uploading file...');
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
      setJobStatus(`Job submitted! You can monitor its status.`);
      // In a real app, we'd start polling jobResult.status_url
      console.log('Job status URL:', jobResult.status_url);

    } catch (err) {
      console.error(err);
      setJobStatus(`Error: ${err.message}`);
    }
  };

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
          onChange={setSelectedPipeline}
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
          {jobStatus && <p style={{ marginTop: '1rem' }}>Status: {jobStatus}</p>}
        </>
      )}
    </main>
  );
} 