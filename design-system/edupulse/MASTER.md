# EduPulse design tokens

This file records the visual decisions behind `web/`. The values here are the same ones declared in `web/src/app/globals.css`; if the two ever disagree, the stylesheet wins and this file needs updating. Page-level overrides would live in `design-system/pages/<page>.md`, but no page currently needs one.

Project: EduPulse
Last revised: 2026-09-16
Stack: Next.js 16, Tailwind 4, shadcn/ui (`base-nova` style, Lucide icons), Recharts

## Direction

The site has two moods that share one token set.

The landing page is editorial: a serif italic headline, a monospace eyebrow above each section, wide measure, numbers presented as typeset figures rather than dashboard tiles. It borrows from magazine layouts, not from SaaS templates.

The app pages (`/overview`, `/predict`, `/leaderboard`, `/explain`, `/fairness`, `/monitoring`) are dense and dark, built from shadcn cards on a sidebar shell. Charts do most of the talking.

The root `<html>` element carries the `dark` class, so dark is the default everywhere. Marketing sections that want a light background add the `light` class to their wrapper and inherit the light token set.

## Colour

All colours are CSS custom properties consumed through Tailwind's `@theme inline` block, so classes such as `bg-primary`, `text-muted-foreground` and `border-risk-moderate/40` resolve to these values.

### Dark (default)

| Token | Value | Used for |
|---|---|---|
| `--background` | `#0f1220` | page background |
| `--foreground` | `#f5f3ff` | body text |
| `--card` | `#181a2b` | cards, popovers |
| `--card-foreground` | `#f8fafc` | text on cards |
| `--primary` | `#8b5cf6` | violet accent: buttons, links, focus ring, active nav |
| `--primary-foreground` | `#0f172a` | text on primary |
| `--secondary` | `#334155` | secondary buttons |
| `--muted` | `#22253a` | subdued surfaces, table stripes |
| `--muted-foreground` | `#9d9fb8` | captions, labels |
| `--accent` | `#22253a` | hover surfaces |
| `--destructive` | `#ef4444` | errors |
| `--border` | `#2a2d45` | hairlines |
| `--input` | `#475569` | form control borders |
| `--ring` | `#8b5cf6` | focus ring |
| `--sidebar` | `#0b0e1a` | app shell sidebar |
| `--sidebar-accent` | `#181a2b` | sidebar hover and active item |

### Light (`.light` wrapper, also the `:root` fallback)

| Token | Value |
|---|---|
| `--background` | `#f4f7fb` |
| `--foreground` | `#0f172a` |
| `--card` | `#ffffff` |
| `--primary` | `#6d28d9` |
| `--primary-foreground` | `#f8fafc` |
| `--secondary` | `#e2e8f0` |
| `--muted` | `#e9eef5` |
| `--muted-foreground` | `#475569` |
| `--destructive` | `#dc2626` |
| `--border`, `--input` | `#d5dde8` |
| `--ring` | `#6d28d9` |

The light primary is one step darker than the dark primary so both pass 4.5:1 against their backgrounds.

### Charts

Five series colours, in the order Recharts picks them up.

| Token | Dark | Light |
|---|---|---|
| `--chart-1` | `#8b5cf6` | `#7c3aed` |
| `--chart-2` | `#38bdf8` | `#38bdf8` |
| `--chart-3` | `#fbbf24` | `#f59e0b` |
| `--chart-4` | `#22c55e` | `#22c55e` |
| `--chart-5` | `#fb7185` | `#f43f5e` |

Chart 1 is the model or "after" series. The mitigation figure rendered by the training pipeline (`reports/figures/`) uses a matplotlib palette of its own (`PALETTE` in `src/edupulse/models/evaluate.py`): grey `#94A3B8` for the global-threshold bars and indigo `#4F46E5` for the per-group bars, so the coloured series reads as the change.

### Risk bands

These map to the `risk_band` field the API returns and are the same in both modes.

| Token | Value | Band |
|---|---|---|
| `--risk-low` | `#22c55e` | low |
| `--risk-moderate` | `#f59e0b` | moderate |
| `--risk-high` | `#f97316` | high |
| `--risk-critical` | `#ef4444` | critical |

The monitoring page reuses moderate for `warn` and critical for `alert`.

## Type

Three families, loaded through `next/font/google` in `web/src/app/layout.tsx` and exposed as CSS variables.

| Role | Family | Variable | Weights |
|---|---|---|---|
| Body and UI | Inter | `--font-inter` | 300 to 700 |
| Display | Playfair Display | `--font-playfair` | 400 to 700, italic included |
| Mono | JetBrains Mono | `--font-jetbrains` | 400 to 600 |

`body` uses `font-sans` (Inter). Two helper classes carry the editorial voice:

- `.font-display`: Playfair Display with `letter-spacing: -0.01em`. Used for the landing headline, section titles, and the large figures on the landing page.
- `.eyebrow`: JetBrains Mono, `0.7rem`, `0.18em` tracking, uppercase. Used for the numbered section labels ("01. The problem") and small metadata lines.

Numbers in the app (metrics, thresholds, probabilities) are set in the mono face so columns align.

## Radius and spacing

`--radius` is `0.625rem`. shadcn derives `--radius-sm` (−4px), `--radius-md` (−2px), `--radius-lg` (equal) and `--radius-xl` (+4px) from it. Spacing follows Tailwind's default 4px scale; there is no custom spacing scale.

## Surfaces

`.glass` is the card treatment on dark pages: the card colour at 88% opacity mixed in oklab, a 12px backdrop blur, and a border at 80% of `--border`. It is applied on top of the shadcn `Card` component rather than replacing it, so `<Card className="glass">` is the usual form.

## Motion

One pattern, implemented without a library.

`.reveal` starts at `opacity: 0; translateY(16px)` and transitions both over 600ms with `cubic-bezier(0.22, 1, 0.36, 1)`. The `Reveal` component in `web/src/components/marketing/reveal.tsx` adds `is-visible` the first time the element enters the viewport (IntersectionObserver, `threshold: 0.1`, bottom margin −10%) and then disconnects. A `delay` prop staggers siblings through `transition-delay`.

Under `prefers-reduced-motion: reduce`, `.reveal` renders in its final state with no transition, and a global rule shortens every animation and transition to 0.01ms. Browsers without IntersectionObserver get the visible state immediately.

App pages do not use scroll reveals; they only have the hover and focus transitions that shadcn ships.

## Rules of thumb

- Icons come from Lucide, never from emoji.
- Text on any surface keeps at least 4.5:1 contrast; check both modes when adding a colour.
- Every interactive element shows a visible focus ring (`outline-ring/50` is set globally).
- Hover effects change colour or shadow, not layout.
- Layouts are checked at 375, 768, 1024 and 1440px, with no horizontal scroll on the narrowest.
- The landing page fetches its headline numbers from the API at request time and falls back to the constants in `web/src/app/(marketing)/page.tsx` when the API is asleep. The swap is silent, so keep those constants equal to the committed model cards.
