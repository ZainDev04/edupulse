"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Activity, ArrowUpRight, Menu, X } from "lucide-react";
import { cn } from "cn";
import { Button } from "@/components/ui/button";

const LINKS = [
  { href: "#problem", label: "The problem" },
  { href: "#results", label: "Results" },
  { href: "#demo", label: "Live demo" },
  { href: "#how", label: "How it works" },
  { href: "#fairness", label: "Fairness" },
];

export function SiteNav() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      className={cn(
        "lp-nav fixed inset-x-0 top-0 z-40 transition-colors duration-300",
        scrolled || open ? "bg-[#0b0e1a]/85 backdrop-blur-md border-b border-white/10" : "bg-transparent",
      )}
    >
      <div className="mx-auto flex h-16 max-w-7xl items-center gap-6 px-5 sm:px-8">
        <Link href="/" className="flex items-center gap-2 font-display text-lg font-semibold text-white">
          <Activity className="size-5 text-primary" aria-hidden="true" />
          EduPulse
        </Link>
        <nav aria-label="Sections" className="ml-auto hidden items-center gap-7 md:flex">
          {LINKS.map((l) => (
            <a key={l.href} href={l.href} className="text-sm text-white/70 transition-colors hover:text-white">
              {l.label}
            </a>
          ))}
        </nav>
        <div className="ml-auto hidden items-center gap-2 md:ml-0 md:flex">
          <Button render={<a href="https://github.com/ZainDev04/edupulse" target="_blank" rel="noreferrer" />} nativeButton={false} variant="ghost" size="sm" className="text-white/80 hover:text-white">
            GitHub <ArrowUpRight className="size-4" aria-hidden="true" />
          </Button>
          <Button render={<Link href="/overview" />} nativeButton={false} size="sm">
            Open the app
          </Button>
        </div>
        <Button
          variant="ghost"
          size="icon"
          className="ml-auto text-white md:hidden"
          aria-label={open ? "Close menu" : "Open menu"}
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
        >
          {open ? <X className="size-5" /> : <Menu className="size-5" />}
        </Button>
      </div>
      {open && (
        <nav aria-label="Sections" className="flex flex-col gap-1 border-t border-white/10 px-5 py-3 md:hidden">
          {LINKS.map((l) => (
            <a key={l.href} href={l.href} onClick={() => setOpen(false)} className="min-h-11 py-2 text-sm text-white/80">
              {l.label}
            </a>
          ))}
          <Link href="/overview" onClick={() => setOpen(false)} className="min-h-11 py-2 text-sm font-medium text-white">
            Open the app
          </Link>
        </nav>
      )}
    </header>
  );
}
