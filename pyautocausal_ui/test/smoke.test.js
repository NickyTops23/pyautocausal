/**
 * @jest-environment jsdom
 */

const fs = require('fs');
const path = require('path');

describe('PyAutoCausal UI smoke test', () => {
  beforeEach(async () => {
    // Load the HTML into jsdom
    const html = fs.readFileSync(path.resolve(__dirname, '../legacy_index.html'), 'utf8');
    document.documentElement.innerHTML = html;

    // Mock fetch pipelines response
    global.fetch = jest.fn(() =>
      Promise.resolve({ ok: true, json: () => Promise.resolve([]) })
    );

    // Require the script (IIFE will execute)
    jest.isolateModules(() => {
      require('../script.js');
    });
  });

  test('initial UI elements present', () => {
    expect(document.getElementById('pipeline-select')).toBeTruthy();
    expect(document.getElementById('file-input')).toBeTruthy();
    expect(document.getElementById('upload-btn')).toBeTruthy();
  });

  test('upload button disabled until conditions met', () => {
    const uploadBtn = document.getElementById('upload-btn');
    expect(uploadBtn.disabled).toBe(true);
  });
}); 