import React, { useState } from 'react';
import usePipelines from './hooks/usePipelines';
import PipelineSelect from './components/PipelineSelect';

export default function App() {
  const { pipelines, loading, error } = usePipelines();
  const [selected, setSelected] = useState(null);

  return (
    <main style={{ padding: '1rem' }}>
      <h1>PyAutoCausal React UI</h1>
      <PipelineSelect
        pipelines={pipelines}
        loading={loading}
        error={error}
        value={selected}
        onChange={setSelected}
      />
      {selected && <pre>Selected: {selected}</pre>}
    </main>
  );
} 