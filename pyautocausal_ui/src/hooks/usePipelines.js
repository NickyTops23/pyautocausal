import { useEffect, useState } from 'react';

export default function usePipelines() {
  const [pipelines, setPipelines] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchPipelines() {
      try {
        const baseUrl = window?.PYAUTOCAUSAL_UI_CONFIG?.apiBaseUrl;
        if (!baseUrl) throw new Error('apiBaseUrl is not defined in config');
        const res = await fetch(`${baseUrl}/pipelines`);
        if (!res.ok) throw new Error(`Failed to fetch pipelines: ${res.status}`);
        const data = await res.json();
        // data is a mapping object → convert to array of objects with name
        const list = Object.entries(data).map(([name, meta]) => ({ name, ...meta }));
        setPipelines(list);
      } catch (err) {
        console.error(err);
        setError(err);
      } finally {
        setLoading(false);
      }
    }
    fetchPipelines();
  }, []);

  return { pipelines, loading, error };
} 