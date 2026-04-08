import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "AMI Predictor",
  description: "Top-3 ensemble predictor for Acute Myocardial Infarction",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
