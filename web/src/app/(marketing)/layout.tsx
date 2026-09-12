import Link from "next/link";
import { Activity } from "lucide-react";
import { SiteNav } from "@/components/marketing/site-nav";

export default function MarketingLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col bg-[#0b0e1a] text-white">
      <SiteNav />
      <main className="flex-1">{children}</main>
      <footer className="border-t border-white/10 bg-[#0b0e1a] px-5 py-12 sm:px-8">
        <div className="mx-auto flex max-w-7xl flex-col gap-8 md:flex-row md:items-start md:justify-between">
          <div className="flex flex-col gap-3">
            <Link href="/" className="flex items-center gap-2 font-display text-lg font-semibold">
              <Activity className="size-5 text-primary" aria-hidden="true" />
              EduPulse
            </Link>
            <p className="max-w-sm text-sm text-white/60">
              Student performance intelligence, built as a portfolio project on the public Students Performance dataset.
              MIT licensed.
            </p>
          </div>
          <nav aria-label="Footer" className="grid grid-cols-2 gap-x-12 gap-y-2 text-sm text-white/70 sm:grid-cols-3">
            <Link href="/overview" className="hover:text-white">App overview</Link>
            <Link href="/predict" className="hover:text-white">Predict</Link>
            <Link href="/leaderboard" className="hover:text-white">Leaderboard</Link>
            <Link href="/explain" className="hover:text-white">Explainability</Link>
            <Link href="/fairness" className="hover:text-white">Fairness</Link>
            <Link href="/monitoring" className="hover:text-white">Monitoring</Link>
            <a href="https://github.com/ZainDev04/edupulse" target="_blank" rel="noreferrer" className="hover:text-white">GitHub</a>
            <a href="https://github.com/ZainDev04/edupulse/blob/main/docs/architecture.md" target="_blank" rel="noreferrer" className="hover:text-white">Architecture notes</a>
            <a href="https://github.com/ZainDev04/edupulse/releases" target="_blank" rel="noreferrer" className="hover:text-white">Releases</a>
          </nav>
        </div>
        <p className="mx-auto mt-10 max-w-7xl text-xs text-white/40">
          Shaikh Muhammad Zain. Originally coursework for Programming for AI at NED University, then rebuilt as a complete system.
        </p>
      </footer>
    </div>
  );
}
