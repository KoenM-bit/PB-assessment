import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    root: "functions",
    include: ["_shared/**/*.test.ts"],
  },
});
