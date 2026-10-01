const { defineConfig, devices } = require('@playwright/test');
const port = Number(process.env.FRONTEND_TEST_PORT || 47891);
const baseURL = `http://127.0.0.1:${port}`;
module.exports = defineConfig({
  testDir: './tests',
  testMatch: '**/*.spec.cjs',
  fullyParallel: false,
  timeout: 15000,
  expect: { timeout: 5000 },
  outputDir: '/tmp/hokiehelp-playwright-results',
  use: { baseURL, trace: 'retain-on-failure' },
  webServer: {
    command: 'node tests/server.cjs',
    cwd: __dirname,
    url: baseURL,
    reuseExistingServer: false,
    timeout: 10000,
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
});
