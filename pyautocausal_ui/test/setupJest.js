// Jest setup for frontend tests

// Polyfill fetch if not defined (Node 18+ has global fetch)
if (typeof fetch === 'undefined') {
  global.fetch = jest.fn(() =>
    Promise.resolve({ ok: true, json: () => Promise.resolve([]) })
  );
}

// Ensure FileReader exists in jsdom (jsdom >=16 provides it)
if (typeof FileReader === 'undefined') {
  global.FileReader = class MockFileReader {
    readAsText() {}
  };
} 