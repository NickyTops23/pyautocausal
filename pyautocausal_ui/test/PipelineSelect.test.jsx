import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import React from 'react';
import App from '../src/App';

beforeAll(() => {
  global.fetch = jest.fn(() =>
    Promise.resolve({
      ok: true,
      json: () =>
        Promise.resolve({
          example_graph: {
            required_columns: [],
            optional_columns: [],
          },
        }),
    })
  );
  global.window.PYAUTOCAUSAL_UI_CONFIG = { apiBaseUrl: 'http://localhost:8000' };
});

afterAll(() => {
  jest.resetAllMocks();
});

test('renders pipeline options', async () => {
  render(<App />);
  // Wait for dropdown to appear with option
  await waitFor(() => expect(screen.getByRole('combobox')).toBeInTheDocument());
  expect(screen.getByText('example_graph')).toBeInTheDocument();
}); 