import type { Metadata } from "next";
import { Fira_Code, Fira_Sans } from "next/font/google";
import "./globals.css";
import { AppShell } from "@/components/app-shell";
import { api } from "@/lib/api";

const firaSans = Fira_Sans({
  variable: "--font-fira-sans",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
});

const firaCode = Fira_Code({
  variable: "--font-fira-code",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: {
    default: "EduPulse",
    template: "%s | EduPulse",
  },
  description:
    "Student performance intelligence: early-warning risk scoring, score prediction, SHAP explanations and a fairness audit.",
};

export const dynamic = "force-dynamic";

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const health = await api.health();
  return (
    <html lang="en" className={`dark ${firaSans.variable} ${firaCode.variable} h-full antialiased`}>
      <body className="min-h-full">
        <AppShell health={health}>{children}</AppShell>
      </body>
    </html>
  );
}
