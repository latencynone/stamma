import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';

// Signalsmith Stretch needs a real OfflineAudioContext + AudioWorklet, which
// don't exist in Node — so these tests run in an actual browser (the
// system's installed Chrome, via Playwright's `channel: 'chrome'`, since
// Playwright's own bundled Chromium build isn't supported on this machine's
// macOS version).
export default defineConfig({
  test: {
    include: ['src/**/*.test.js'],
    browser: {
      enabled: true,
      provider: playwright({ launchOptions: { channel: 'chrome' } }),
      headless: true,
      instances: [{ browser: 'chromium' }],
    },
  },
});
