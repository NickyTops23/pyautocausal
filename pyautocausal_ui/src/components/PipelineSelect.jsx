import React from 'react';

export default function PipelineSelect({ pipelines, loading, error, value, onChange }) {
  if (loading) return <p>Loading pipelines…</p>;
  if (error) return <p style={{ color: 'red' }}>Error: {error.message}</p>;

  return (
    <label style={{ display: 'block', marginBottom: '1rem' }}>
      Choose pipeline:{' '}
      <select
        value={value || ''}
        onChange={(e) => onChange(e.target.value || null)}
      >
        <option value="">-- Select a pipeline --</option>
        {pipelines.map((p) => (
          <option key={p.name} value={p.name}>
            {p.display_name || p.name}
          </option>
        ))}
      </select>
    </label>
  );
} 