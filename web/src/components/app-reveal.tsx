"use client";

import { useEffect, useRef } from "react";
import { useSearchParams } from "next/navigation";

const STAGGER_MS = 70;
const KEYFRAMES: Keyframe[] = [
  { opacity: 0, transform: "translateY(18px)" },
  { opacity: 1, transform: "none" },
];
const ID = "ep-reveal";
const TIMING: KeyframeAnimationOptions = {
  id: ID,
  duration: 650,
  easing: "cubic-bezier(0.22, 1, 0.36, 1)",
  fill: "both",
};

/** A section reveals as one block unless it is, or directly holds, a
 *  data-reveal="items" grid, whose children then cascade one by one. */
function targets(section: Element): HTMLElement[] {
  if (section.matches("[data-reveal=items]")) return Array.from(section.children) as HTMLElement[];
  if (!section.querySelector(":scope > [data-reveal=items]")) return [section as HTMLElement];
  return (Array.from(section.children) as HTMLElement[]).flatMap((child) =>
    child.matches("[data-reveal=items]") ? (Array.from(child.children) as HTMLElement[]) : [child],
  );
}

/**
 * Wraps an app page and reveals its top-level sections: the ones already in
 * view rise in with a short stagger, the rest wait until they scroll into
 * view. Animations run through the Web Animations API rather than classes,
 * so the server markup is left as React rendered it (the page streams in
 * behind the loading skeleton and hydrates afterwards) and nothing is
 * hidden before JavaScript runs. When the query string changes (the task
 * picker), the page re-renders in place, so everything below the hero is
 * replayed. Reduced motion disables the effect.
 */
export function AppReveal({ children }: { children: React.ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const query = useSearchParams().toString();
  const replay = useRef<(() => void) | null>(null);
  const lastQuery = useRef(query);

  useEffect(() => {
    const root = ref.current;
    if (!root || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const seen = new WeakSet<Element>();
    const running = new WeakMap<Element, Animation>();

    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (!e.isIntersecting) continue;
          running.get(e.target)?.play();
          io.unobserve(e.target);
        }
      },
      { rootMargin: "0px 0px -8% 0px", threshold: 0.05 },
    );

    const reveal = (el: HTMLElement, i: number, inView: () => number) => {
      // One reveal per element: drop any earlier one, whichever pass made it
      for (const a of el.getAnimations()) if (a.id === ID) a.cancel();
      io.unobserve(el);
      if (el.getBoundingClientRect().top < window.innerHeight) {
        running.set(el, el.animate(KEYFRAMES, { ...TIMING, delay: Math.min(inView(), 8) * STAGGER_MS }));
      } else {
        // Held on the first frame (hidden) until it scrolls into view;
        // items of one group keep their cascade when they arrive together
        const a = el.animate(KEYFRAMES, { ...TIMING, delay: Math.min(i, 8) * STAGGER_MS });
        a.pause();
        running.set(el, a);
        io.observe(el);
      }
    };

    const scan = (skipFirst = false) => {
      const page = root.firstElementChild;
      if (!page) return;
      let order = 0;
      const next = () => order++;
      Array.from(page.children).forEach((section, s) => {
        if (skipFirst && s === 0) return;
        if (seen.has(section)) return;
        seen.add(section);
        targets(section).forEach((el, i) => reveal(el, i, next));
      });
    };

    replay.current = () => {
      const page = root.firstElementChild;
      if (!page) return;
      Array.from(page.children).forEach((section, s) => s > 0 && seen.delete(section));
      scan(true);
    };

    scan();
    const mo = new MutationObserver(() => scan());
    mo.observe(root, { childList: true, subtree: true });
    return () => {
      io.disconnect();
      mo.disconnect();
      replay.current = null;
    };
  }, []);

  useEffect(() => {
    if (lastQuery.current === query) return;
    lastQuery.current = query;
    replay.current?.();
  }, [query]);

  return (
    <div ref={ref} className="ep-page">
      {children}
    </div>
  );
}
