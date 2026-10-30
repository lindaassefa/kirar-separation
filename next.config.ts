import type { NextConfig } from "next";
import path from "path";
import { fileURLToPath } from "url";

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  allowedDevOrigins: ["127.0.0.1"],
  outputFileTracingExcludes: {
    "/*": ["./separator/.venv/**", "./separator/.modal-venv/**", "./separator/runs/**", "./separator/models/**"],
  },
  turbopack: {
    root: path.dirname(fileURLToPath(import.meta.url)),
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
