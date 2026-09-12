import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  // Next 16 blocks dev resources from origins other than the bound host; without this,
  // opening http://127.0.0.1:3000 during `next dev` renders the page but never hydrates.
  allowedDevOrigins: ["127.0.0.1"],
};

export default nextConfig;
