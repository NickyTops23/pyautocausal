import '@testing-library/jest-dom'; 

if (!global.window) global.window = {};
if (!window.PYAUTOCAUSAL_UI_CONFIG) {
  window.PYAUTOCAUSAL_UI_CONFIG = { apiBaseUrl: 'http://localhost:8000' };
} 