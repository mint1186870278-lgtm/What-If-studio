import { defineConfig } from "vite";

export default defineConfig({
  server: {
    host: "127.0.0.1",
    port: 5180,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://localhost:8010",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://localhost:8010",
        ws: true,
      },
    }
  },
  build: {
    outDir: "dist",
    emptyOutDir: true
  }
});
