import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ command }) => ({
  plugins: [react()],
  // FastAPI mounts the built frontend under `/app`, so asset URLs must be relative to that.
  // For dev server we keep it as `/` to avoid 404s.
  base: command === "build" ? "/app/" : "/",
}));

