import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the interface runs on :5173 and forwards /api to the engine server in WSL.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  // A local app loaded from disk; one bundle is fine.
  build: { chunkSizeWarningLimit: 1500 },
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8765" },
  },
});
