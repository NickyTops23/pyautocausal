import React from 'react';

export default function ColumnMapping({ csvHeaders, pipeline, onMappingChange }) {
  // The API returns an array of strings; transform them into objects
  // that the component expects: { id: 'name', label: 'name' }
  const transform = (cols = []) =>
    cols.map((col) => (typeof col === 'string' ? { id: col, label: col } : col));

  const required_columns = transform(pipeline.required_columns);
  const optional_columns = transform(pipeline.optional_columns);

  const handleSelectChange = (pipelineColumn, csvColumn) => {
    onMappingChange(pipelineColumn, csvColumn);
  };

  const renderMappingRow = (pCol, isRequired) => (
    <tr key={pCol.id}>
      <td>
        {pCol.label || pCol.id}
        {isRequired && <span style={{ color: 'red' }}>*</span>}
      </td>
      <td>
        <select
          onChange={(e) => handleSelectChange(pCol.id, e.target.value)}
          required={isRequired}
        >
          <option value="">-- Select column --</option>
          {csvHeaders.map((header) => (
            <option key={header} value={header}>
              {header}
            </option>
          ))}
        </select>
      </td>
    </tr>
  );

  return (
    <div>
      <h3>Column Mapping</h3>
      <p>Map columns from your CSV to the pipeline's expected inputs.</p>
      <table>
        <thead>
          <tr>
            <th>Pipeline Column</th>
            <th>Your CSV Column</th>
          </tr>
        </thead>
        <tbody>
          {required_columns.map((pCol) => renderMappingRow(pCol, true))}
          {optional_columns.map((pCol) => renderMappingRow(pCol, false))}
        </tbody>
      </table>
    </div>
  );
} 