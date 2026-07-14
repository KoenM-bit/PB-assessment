import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    root: "netlify/functions",
    include: ["_shared/**/*.test.ts"],
  },
});
