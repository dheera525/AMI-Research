/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Static export: `next build` writes a fully static site to ./out
  // which FastAPI then serves alongside the /predict API.
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,
};

module.exports = nextConfig;
