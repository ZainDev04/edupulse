"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import {
  Activity,
  BarChart3,
  ExternalLink,
  LayoutDashboard,
  Menu,
  Scale,
  Sparkles,
  Target,
  X,
} from "lucide-react";
import { cn } from "cn";
import type { Health } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

const NAV = [
  { href: "/overview", label: "Overview", icon: LayoutDashboard },
  { href: "/predict", label: "Predict", icon: Target },
  { href: "/leaderboard", label: "Leaderboard", icon: BarChart3 },
  { href: "/explain", label: "Explainability", icon: Sparkles },
  { href: "/fairness", label: "Fairness", icon: Scale },
  { href: "/monitoring", label: "Monitoring", icon: Activity },
] as const;

export function AppShell({ health, children }: { health: Health | null; children: React.ReactNode }) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const online = health?.status === "ok";
  const trained = health ? Object.values(health.models).filter(Boolean).length : 0;

  const nav = (
    <nav aria-label="Primary" className="flex flex-col gap-1">
      {NAV.map(({ href, label, icon: Icon }) => {
        const active = pathname.startsWith(href);
        return (
          <Link
            key={href}
            href={href}
            onClick={() => setOpen(false)}
            aria-current={active ? "page" : undefined}
            className={cn(
              "flex min-h-11 items-center gap-3 rounded-md px-3 text-sm font-medium transition-colors duration-150",
              "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground focus-visible:outline-2 focus-visible:outline-ring",
              active ? "bg-sidebar-accent text-sidebar-accent-foreground" : "text-muted-foreground",
            )}
          >
            <Icon className="size-4 shrink-0" aria-hidden="true" />
            {label}
          </Link>
        );
      })}
    </nav>
  );

  const status = (
    <div className="flex items-center gap-2 text-xs text-muted-foreground">
      <span
        className={cn("inline-block size-2 rounded-full", online ? "bg-primary" : "bg-destructive")}
        aria-hidden="true"
      />
      {online ? (
        <span className="whitespace-nowrap">
          API online<span className="hidden min-[400px]:inline">, {trained}/3 models</span>
        </span>
      ) : (
        "API offline"
      )}
    </div>
  );

  return (
    <div className="flex min-h-screen">
      <aside className="hidden w-60 shrink-0 flex-col border-r border-sidebar-border bg-sidebar px-4 py-5 lg:flex">
        <Link href="/" className="mb-6 flex items-center gap-2 px-2 font-display text-lg font-semibold">
          <Activity className="size-5 text-primary" aria-hidden="true" />
          EduPulse
        </Link>
        {nav}
        <div className="mt-auto flex flex-col gap-3 px-2 pt-6">
          {status}
          <a
            href="https://github.com/ZainDev04/edupulse"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground"
          >
            <ExternalLink className="size-3.5" aria-hidden="true" />
            ZainDev04/edupulse
          </a>
          {health && <Badge variant="outline">v{health.version}</Badge>}
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-2 border-b border-border px-3 py-3 sm:gap-3 sm:px-4 lg:hidden">
          <Button
            variant="ghost"
            size="icon"
            aria-label={open ? "Close navigation" : "Open navigation"}
            aria-expanded={open}
            onClick={() => setOpen((v) => !v)}
          >
            {open ? <X className="size-5" /> : <Menu className="size-5" />}
          </Button>
          <Link href="/" className="flex items-center gap-2 font-display text-lg font-semibold">
            <Activity className="size-5 text-primary" aria-hidden="true" />
            EduPulse
          </Link>
          <div className="ml-auto">{status}</div>
        </header>
        {open && (
          <div className="border-b border-border bg-sidebar px-4 py-3 lg:hidden">{nav}</div>
        )}
        <main className="flex-1 px-4 py-6 sm:px-6 lg:px-8">{children}</main>
      </div>
    </div>
  );
}
