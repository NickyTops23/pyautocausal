import { useEffect, useState } from 'react';

export default function usePipelines() {
  const [pipelines, setPipelines] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchPipelines() {
      try {
        // Resolve API base URL with graceful fall-backs
        let baseUrl = window?.PYAUTOCAUSAL_UI_CONFIG?.apiBaseUrl;
        if (!baseUrl) baseUrl = import.meta?.env?.VITE_API_BASE_URL;
        if (!baseUrl) baseUrl = `${window.location.origin.replace(/\/$/, '')}/api`;
        if (!baseUrl) {
          throw new Error(
            'Unable to determine API base URL. Please set it in config.js or as VITE_API_BASE_URL'
          );
        }
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