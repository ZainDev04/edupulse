import type { Metadata } from "next";
import { Inter, Inter_Tight, JetBrains_Mono, Playfair_Display } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
});

// Display face for the app pages (see .ep-headline in globals.css)
const interTight = Inter_Tight({
  variable: "--font-inter-tight",
  subsets: ["latin"],
  weight: ["500", "600", "700"],
});

const playfair = Playfair_Display({
  variable: "--font-playfair",
  subsets: ["latin"],
  style: ["normal", "italic"],
  weight: ["400", "500", "600", "700"],
});

const jetbrains = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: {
    default: "EduPulse",
    template: "%s | EduPulse",
  },
  description:
    "Student performance intelligence: early-warning risk scoring, score prediction with conformal intervals, SHAP explanations, a fairness audit with mitigation, and drift monitoring.",
};

export const dynamic = "force-dynamic";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`dark ${inter.variable} ${interTight.variable} ${playfair.variable} ${jetbrains.variable} h-full antialiased`}
    >
      <body className="min-h-full">{children}</body>
    </html>
  );
}
