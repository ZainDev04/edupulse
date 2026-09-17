"use client";

import { useEffect, useSyncExternalStore } from "react";
import { Moon, Sun } from "lucide-react";

export const THEME_KEY = "edupulse-theme";
type Theme = "dark" | "light";

/**
 * Runs before hydration on app pages (rendered by the app layout) so the
 * stored theme is in place before the first paint. Light is the default.
 */
export const THEME_SCRIPT = `try{var t=localStorage.getItem(${JSON.stringify(THEME_KEY)});var r=document.documentElement;r.classList.remove("dark","light");r.classList.add(t==="dark"?"dark":"light")}catch(e){document.documentElement.classList.replace("dark","light")}`;

function apply(theme: Theme) {
  const root = document.documentElement;
  root.classList.remove("dark", "light");
  root.classList.add(theme);
}

function stored(): Theme {
  try {
    return window.localStorage.getItem(THEME_KEY) === "dark" ? "dark" : "light";
  } catch {
    return "light";
  }
}

/**
 * Mounted once in the app layout. The root layout renders the html element
 * with the dark class for the landing page; this component keeps the app
 * pages on the stored theme and puts dark back when the app group unmounts.
 */
export function AppTheme() {
  useEffect(() => {
    apply(stored());
    return () => apply("dark");
  }, []);
  return null;
}

function subscribe(onChange: () => void) {
  const observer = new MutationObserver(onChange);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
  return () => observer.disconnect();
}
const current = (): Theme => (document.documentElement.classList.contains("dark") ? "dark" : "light");

/** Light/dark switch shown in every page hero. State is read from the html class. */
export function ThemeToggle() {
  const theme = useSyncExternalStore(subscribe, current, () => "light" as Theme);

  function toggle() {
    const next: Theme = theme === "dark" ? "light" : "dark";
    apply(next);
    try {
      window.localStorage.setItem(THEME_KEY, next);
    } catch {
      // storage blocked; the choice lasts for this visit only
    }
  }

  const light = theme === "light";
  return (
    <button
      type="button"
      onClick={toggle}
      aria-pressed={!light}
      aria-label={light ? "Switch to dark mode" : "Switch to light mode"}
      title={light ? "Dark mode" : "Light mode"}
      className="inline-flex h-10 items-center gap-2 rounded-full border border-white/20 bg-white/10 px-3 text-xs font-medium text-white backdrop-blur transition-colors hover:bg-white/20 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-white/50"
    >
      {light ? <Moon className="size-4" aria-hidden="true" /> : <Sun className="size-4" aria-hidden="true" />}
      <span className="hidden sm:inline">{light ? "Dark" : "Light"}</span>
    </button>
  );
}
