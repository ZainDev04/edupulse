// Shared building blocks for the app pages. The look follows a splash-page
// pattern: dark aurora hero, glow tiles with gradient numerals, pill chips
// and two-tone section headings. Styles live under .ep-* in globals.css.
import type { LucideIcon } from "lucide-react";
import { cn } from "cn";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ThemeToggle } from "@/components/theme";

export type Tone = "violet" | "blue" | "green" | "amber" | "rose";

/** Page header. Stays dark in both themes, so everything inside uses white-based classes. */
export function PageHero({
  eyebrows,
  title,
  description,
  children,
  compact,
}: {
  eyebrows?: React.ReactNode[];
  title: React.ReactNode;
  description: React.ReactNode;
  /** Actions or a task picker, rendered under the description. */
  children?: React.ReactNode;
  compact?: boolean;
}) {
  return (
    <section
      className={cn(
        "ep-aurora relative overflow-hidden rounded-3xl border border-white/10 px-5 text-center sm:rounded-[28px] sm:px-10",
        compact ? "pt-14 pb-8 sm:py-12" : "pt-16 pb-10 sm:py-20",
      )}
    >
      <div className="absolute top-3 right-3 sm:top-4 sm:right-4">
        <ThemeToggle />
      </div>
      <div className="relative mx-auto flex max-w-3xl flex-col items-center gap-4 sm:gap-5">
        {eyebrows && eyebrows.length > 0 && (
          <div className="flex flex-wrap items-center justify-center gap-2">
            {eyebrows.map((e, i) => (
              <span
                key={i}
                className={cn(
                  "rounded-full px-3 py-1 text-xs font-medium",
                  i === 0
                    ? "bg-sky-500/90 font-semibold text-white"
                    : "border border-white/20 bg-white/10 text-white/90 backdrop-blur",
                )}
              >
                {e}
              </span>
            ))}
          </div>
        )}
        <h1
          className={cn(
            "ep-headline text-balance text-white",
            compact ? "text-3xl sm:text-5xl" : "text-[2rem] min-[400px]:text-4xl sm:text-6xl",
          )}
        >
          {title}
        </h1>
        <p className="max-w-2xl text-sm text-white/80 min-[400px]:text-base sm:text-lg">{description}</p>
        {children && <div className="flex flex-wrap items-center justify-center gap-3">{children}</div>}
      </div>
    </section>
  );
}

export const HERO_BUTTON = "h-11 rounded-full px-5 text-sm font-medium focus-visible:ring-4 focus-visible:ring-white/50";
export const HERO_BUTTON_PRIMARY = `${HERO_BUTTON} bg-violet-500 text-white hover:bg-violet-400`;
export const HERO_BUTTON_SECONDARY = `${HERO_BUTTON} bg-white text-slate-900 hover:bg-white/90`;

export function Eyebrow({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded-full border border-border bg-card px-3 py-1 text-[11px] font-medium tracking-[0.18em] text-muted-foreground uppercase">
      {children}
    </span>
  );
}

/** Gradient-filled words inside a heading. */
export function Accent({ children }: { children: React.ReactNode }) {
  return <span className="ep-gradient-text">{children}</span>;
}

export function SectionHeading({
  id,
  eyebrow,
  title,
  description,
}: {
  id?: string;
  eyebrow: string;
  title: React.ReactNode;
  description?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center gap-3 text-center">
      <Eyebrow>{eyebrow}</Eyebrow>
      <h2 id={id} className="ep-headline text-3xl sm:text-4xl">
        {title}
      </h2>
      {description && <p className="max-w-xl text-sm text-muted-foreground">{description}</p>}
    </div>
  );
}

/** Stat tile with a coloured glow and a gradient numeral. */
export function Tile({
  tone,
  icon: Icon,
  value,
  label,
  hint,
}: {
  tone: Tone;
  icon: LucideIcon;
  value: string;
  label: string;
  hint?: string;
}) {
  const long = value.length > 8;
  return (
    <div className={`ep-tile ep-tile-${tone} flex min-h-40 flex-col justify-between rounded-3xl p-6`}>
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm font-medium text-foreground/80">{label}</span>
        <span className="flex size-9 shrink-0 items-center justify-center rounded-full bg-foreground/10">
          <Icon className="size-4 text-foreground" aria-hidden="true" />
        </span>
      </div>
      <div>
        <div className={cn("ep-headline ep-gradient-text tabular-nums", long ? "text-2xl sm:text-3xl" : "text-4xl sm:text-5xl")}>
          {value}
        </div>
        {hint && <div className="mt-1 text-xs text-muted-foreground">{hint}</div>}
      </div>
    </div>
  );
}

/** Card with the tile treatment and a tight display title. */
export function Panel({
  title,
  description,
  tone,
  className,
  contentClassName,
  children,
}: {
  title?: React.ReactNode;
  description?: React.ReactNode;
  tone?: Tone;
  className?: string;
  contentClassName?: string;
  children: React.ReactNode;
}) {
  return (
    <Card className={cn("ep-tile rounded-3xl", tone && `ep-tile-${tone}`, className)}>
      {(title || description) && (
        <CardHeader>
          {title && <CardTitle className="ep-headline text-xl">{title}</CardTitle>}
          {description && <CardDescription>{description}</CardDescription>}
        </CardHeader>
      )}
      <CardContent className={contentClassName}>{children}</CardContent>
    </Card>
  );
}

/** Pill chip for a labelled figure. */
export function Chip({
  icon: Icon,
  children,
  value,
  tone = "default",
}: {
  icon?: LucideIcon;
  children: React.ReactNode;
  value?: React.ReactNode;
  tone?: "default" | "warn" | "alert";
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 rounded-full border py-1.5 pr-3 pl-2.5 text-sm",
        tone === "default" && "border-border bg-background/60",
        tone === "warn" && "border-amber-500/40 bg-amber-500/10",
        tone === "alert" && "border-destructive/40 bg-destructive/10",
      )}
    >
      {Icon && (
        <Icon
          className={cn(
            "size-4",
            tone === "default" && "text-lime-600 dark:text-lime-400",
            tone === "warn" && "text-amber-600 dark:text-amber-400",
            tone === "alert" && "text-destructive",
          )}
          aria-hidden="true"
        />
      )}
      <span>{children}</span>
      {value !== undefined && (
        <span className="ep-gradient-text font-mono text-sm font-semibold tabular-nums">{value}</span>
      )}
    </span>
  );
}
