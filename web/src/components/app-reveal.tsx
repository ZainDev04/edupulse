"use client";

import { useEffect, useRef } from "react";

const STAGGER_MS = 70;
const KEYFRAMES: Keyframe[] = [
  { opacity: 0, transform: "translateY(18px)" },
  { opacity: 1, transform: "none" },
];
const TIMING: KeyframeAnimationOptions = {
  duration: 650,
  easing: "cubic-bezier(0.22, 1, 0.36, 1)",
  fill: "both",
};

/**
 * Wraps an app page and reveals its top-level sections: the ones already in
 * view rise in with a short stagger, the rest wait until they scroll into
 * view. Animations run through the Web Animations API rather than classes,
 * so the server markup is left as React rendered it (the page streams in
 * behind the loading skeleton and hydrates afterwards) and nothing is
 * hidden before JavaScript runs. Reduced motion disables the effect.
 */
export function AppReveal({ children }: { children: React.ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const root = ref.current;
    if (!root || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const seen = new WeakSet<Element>();
    const pending = new WeakMap<Element, Animation>();

    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (!e.isIntersecting) continue;
          pending.get(e.target)?.play();
          pending.delete(e.target);
          io.unobserve(e.target);
        }
      },
      { rootMargin: "0px 0px -8% 0px", threshold: 0.05 },
    );

    const scan = () => {
      const page = root.firstElementChild;
      if (!page) return;
      let order = 0;
      for (const el of Array.from(page.children) as HTMLElement[]) {
        if (seen.has(el)) continue;
        seen.add(el);
        if (el.getBoundingClientRect().top < window.innerHeight) {
          el.animate(KEYFRAMES, { ...TIMING, delay: Math.min(order, 5) * STAGGER_MS });
          order += 1;
        } else {
          // Held on the first frame (hidden) until the section scrolls into view
          const a = el.animate(KEYFRAMES, TIMING);
          a.pause();
          pending.set(el, a);
          io.observe(el);
        }
      }
    };

    scan();
    const mo = new MutationObserver(scan);
    mo.observe(root, { childList: true, subtree: true });
    return () => {
      io.disconnect();
      mo.disconnect();
    };
  }, []);

  return (
    <div ref={ref} className="ep-page">
      {children}
    </div>
  );
}
