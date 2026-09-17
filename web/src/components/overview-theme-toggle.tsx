"use client";

import { useEffect, useState } from "react";
import { Moon, Sun } from "lucide-react";

export const OVERVIEW_THEME_KEY = "edupulse-overview-theme";
type Theme = "dark" | "light";

function apply(theme: Theme) {
  const root = document.documentElement;
  root.classList.remove("dark", "light");
  root.classList.add(theme);
}

/**
 * Light/dark switch for the overview page only. The app is server-rendered
 * with the dark palette; this swaps the html class while the page is
 * mounted and restores dark on unmount so the other pages are untouched.
 * The inline script in the page applies the stored choice before hydration.
 */
export function OverviewThemeToggle() {
  const [theme, setTheme] = useState<Theme>("dark");

  useEffect(() => {
    let stored: Theme = "dark";
    try {
      if (window.localStorage.getItem(OVERVIEW_THEME_KEY) === "light") stored = "light";
    } catch {
      // storage blocked; keep the default
    }
    setTheme(stored);
    apply(stored);
    return () => apply("dark");
  }, []);

  function toggle() {
    const next: Theme = theme === "dark" ? "light" : "dark";
    setTheme(next);
    apply(next);
    try {
      window.localStorage.setItem(OVERVIEW_THEME_KEY, next);
    } catch {
      // storage blocked; the choice lasts for this visit only
    }
  }

  const light = theme === "light";
  return (
    <button
      type="button"
      onClick={toggle}
      aria-pressed={light}
      aria-label={light ? "Switch overview to dark mode" : "Switch overview to light mode"}
      title={light ? "Dark mode" : "Light mode"}
      className="inline-flex h-10 items-center gap-2 rounded-full border border-white/20 bg-white/10 px-3 text-xs font-medium text-white backdrop-blur transition-colors hover:bg-white/20 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-white/50"
    >
      {light ? <Moon className="size-4" aria-hidden="true" /> : <Sun className="size-4" aria-hidden="true" />}
      <span className="hidden sm:inline">{light ? "Dark" : "Light"}</span>
    </button>
  );
}
