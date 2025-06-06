import React from 'react';

export default function FileInput({ onFileSelect, disabled }) {
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type === "text/csv") {
      onFileSelect(file);
    } else {
      onFileSelect(null);
      // Optionally, provide feedback to the user about invalid file type
      alert("Please select a CSV file.");
    }
  };

  return (
    <div>
      <label htmlFor="file-upload">Upload CSV:</label>
      <input
        id="file-upload"
        type="file"
        accept=".csv"
        onChange={handleFileChange}
        disabled={disabled}
      />
    </div>
  );
} 