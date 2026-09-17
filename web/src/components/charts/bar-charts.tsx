"use client";

import { useSyncExternalStore } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const AXIS = { fill: "var(--muted-foreground)", fontSize: 12 };
const GRID = "var(--border)";

/** Viewport flags: below the 768px tablet boundary the charts trade label
 *  room for bar room; below 400px (iPhone SE class) labels wrap harder. */
const COMPACT = "(max-width: 767px)";
const NARROW = "(max-width: 399px)";
function useMedia(query: string) {
  return useSyncExternalStore(
    (onChange) => {
      const mql = window.matchMedia(query);
      mql.addEventListener("change", onChange);
      return () => mql.removeEventListener("change", onChange);
    },
    () => window.matchMedia(query).matches,
    () => false,
  );
}
const useCompact = () => useMedia(COMPACT);
const useNarrow = () => useMedia(NARROW);

/** Greedy word wrap for tick labels; underscores read as spaces and, when
 *  asked, a slash is a break opportunity too ("free/reduced"). */
function wrapLabel(raw: string, max: number, lines = 2, breakSlash = false): string[] {
  const text = raw.replace(/_/g, " ").replace(/\//g, breakSlash ? "/ " : "/");
  const words = text.split(" ");
  const out: string[] = [];
  for (const w of words) {
    const last = out[out.length - 1];
    if (last !== undefined && (last + " " + w).length <= max) out[out.length - 1] = last + " " + w;
    else out.push(w);
  }
  if (out.length > lines) return [...out.slice(0, lines - 1), out.slice(lines - 1).join(" ")];
  return out;
}

type TickProps = { x?: number; y?: number; payload?: { value: string | number } };

/** Y-axis category tick that wraps long names on narrow screens. */
function WrappedYTick({ x = 0, y = 0, payload, max }: TickProps & { max: number }) {
  const lines = wrapLabel(String(payload?.value ?? ""), max);
  const start = lines.length > 1 ? -2 : 4;
  return (
    <text x={x} y={y} textAnchor="end" fill="var(--muted-foreground)" fontSize={11}>
      {lines.map((l, i) => (
        <tspan key={i} x={x} dy={i === 0 ? start : 12}>
          {l}
        </tspan>
      ))}
    </text>
  );
}

/** X-axis category tick that wraps at spaces on narrow screens. */
function WrappedXTick({ x = 0, y = 0, payload, max, breakSlash }: TickProps & { max: number; breakSlash?: boolean }) {
  const lines = wrapLabel(String(payload?.value ?? ""), max, 3, breakSlash);
  return (
    <text x={x} y={y} textAnchor="middle" fill="var(--muted-foreground)" fontSize={10}>
      {lines.map((l, i) => (
        <tspan key={i} x={x} dy={i === 0 ? 12 : 11}>
          {l}
        </tspan>
      ))}
    </text>
  );
}
const PALETTE = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)", "var(--chart-5)"];

export type Fmt = "fixed3" | "fixed2" | "pct";
const FORMATS: Record<Fmt, (v: number) => string> = {
  fixed3: (v) => v.toFixed(3),
  fixed2: (v) => v.toFixed(2),
  pct: (v) => `${(v * 100).toFixed(0)}%`,
};

const tooltipStyle = {
  backgroundColor: "var(--popover)",
  border: "1px solid var(--border)",
  borderRadius: 8,
  color: "var(--popover-foreground)",
  fontSize: 12,
};

/** Ranked horizontal bars with optional error bars; used for leaderboards and importance. */
export function RankedBars({
  data,
  valueKey,
  errorKey,
  labelKey = "name",
  colorKey,
  colorMap,
  format: fmtKey = "fixed3",
  domain,
  height,
  reference,
}: {
  data: Record<string, string | number>[];
  valueKey: string;
  errorKey?: string;
  labelKey?: string;
  colorKey?: string;
  /** Explicit colour per colorKey value; falls back to the palette order. */
  colorMap?: Record<string, string>;
  format?: Fmt;
  domain?: [number, number];
  height?: number;
  reference?: { value: number; label: string };
}) {
  const format = FORMATS[fmtKey];
  const compact = useCompact();
  const categories = colorKey ? Array.from(new Set(data.map((d) => String(d[colorKey])))) : [];
  const h = height ?? Math.max(220, (compact ? 38 : 34) * data.length + 40);
  return (
    <ResponsiveContainer width="100%" height={h}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: compact ? 40 : 48, bottom: 4, left: compact ? 0 : 8 }}>
        <CartesianGrid horizontal={false} stroke={GRID} strokeDasharray="3 3" />
        <XAxis type="number" domain={domain ?? [0, "auto"]} tick={{ ...AXIS, fontSize: compact ? 11 : 12 }} axisLine={false} tickLine={false} tickFormatter={(v) => format(Number(v))} />
        <YAxis
          type="category"
          dataKey={labelKey}
          width={compact ? 96 : 170}
          tick={compact ? <WrappedYTick max={14} /> : AXIS}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          cursor={{ fill: "var(--muted)", opacity: 0.4 }}
          contentStyle={tooltipStyle}
          formatter={(v) => format(Number(v))}
        />
        {reference && (
          <ReferenceLine
            x={reference.value}
            stroke="var(--muted-foreground)"
            strokeDasharray="4 4"
            label={{ value: reference.label, fill: "var(--muted-foreground)", fontSize: 11, position: "top" }}
          />
        )}
        <Bar
          dataKey={valueKey}
          radius={[0, 4, 4, 0]}
          isAnimationActive={false}
          label={{ position: "right", fill: "var(--foreground)", fontSize: 11, formatter: (v: unknown) => format(Number(v)) }}
        >
          {data.map((d, i) => (
            <Cell
              key={i}
              fill={
                colorKey
                  ? (colorMap?.[String(d[colorKey])] ?? PALETTE[categories.indexOf(String(d[colorKey])) % PALETTE.length])
                  : PALETTE[0]
              }
            />
          ))}
          {errorKey && <ErrorBar dataKey={errorKey} width={4} strokeWidth={1.5} stroke="var(--muted-foreground)" direction="x" />}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Signed contributions (SHAP) with colour by sign. */
export function ContributionBars({ data }: { data: { feature: string; contribution: number; value: string | number | null }[] }) {
  const max = Math.ceil(Math.max(...data.map((d) => Math.abs(d.contribution)), 0.01) * 10) / 10;
  const compact = useCompact();
  return (
    <ResponsiveContainer width="100%" height={Math.max(200, 40 * data.length + 30)}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: compact ? 12 : 40, bottom: 4, left: compact ? 0 : 8 }}>
        <CartesianGrid horizontal={false} stroke={GRID} strokeDasharray="3 3" />
        <XAxis type="number" domain={[-max, max]} ticks={[-max, -max / 2, 0, max / 2, max]} tick={{ ...AXIS, fontSize: compact ? 11 : 12 }} axisLine={false} tickLine={false} tickFormatter={(v) => Number(v).toFixed(1)} />
        <YAxis
          type="category"
          dataKey="feature"
          width={compact ? 96 : 180}
          tick={compact ? <WrappedYTick max={14} /> : AXIS}
          axisLine={false}
          tickLine={false}
        />
        <ReferenceLine x={0} stroke="var(--muted-foreground)" />
        <Tooltip
          cursor={{ fill: "var(--muted)", opacity: 0.4 }}
          contentStyle={tooltipStyle}
          formatter={(v, _n, item) => [`${Number(v).toFixed(3)} (value: ${String(item.payload.value)})`, "contribution"]}
        />
        <Bar dataKey="contribution" radius={4} isAnimationActive={false}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.contribution >= 0 ? "var(--risk-critical)" : "var(--chart-1)"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Grouped vertical bars: one group per category, one bar per metric. */
export function GroupedBars({
  data,
  metrics,
  format: fmtKey = "fixed2",
}: {
  data: Record<string, string | number>[];
  metrics: { key: string; label: string }[];
  format?: Fmt;
}) {
  const format = FORMATS[fmtKey];
  const compact = useCompact();
  const narrow = useNarrow();
  return (
    <ResponsiveContainer width="100%" height={compact ? 300 : 320}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 8, left: compact ? -16 : 0 }}>
        <CartesianGrid vertical={false} stroke={GRID} strokeDasharray="3 3" />
        <XAxis
          dataKey="group"
          tick={compact ? <WrappedXTick max={narrow ? 8 : 12} breakSlash={narrow} /> : AXIS}
          height={compact ? 44 : 30}
          axisLine={false}
          tickLine={false}
          interval={0}
        />
        <YAxis tick={{ ...AXIS, fontSize: compact ? 11 : 12 }} axisLine={false} tickLine={false} domain={[0, 1]} />
        <Tooltip cursor={{ fill: "var(--muted)", opacity: 0.4 }} contentStyle={tooltipStyle} formatter={(v) => format(Number(v))} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {metrics.map((m, i) => (
          <Bar key={m.key} dataKey={m.key} name={m.label} fill={PALETTE[i % PALETTE.length]} radius={[3, 3, 0, 0]} isAnimationActive={false} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
