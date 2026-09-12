"use client";

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
  const categories = colorKey ? Array.from(new Set(data.map((d) => String(d[colorKey])))) : [];
  const h = height ?? Math.max(220, 34 * data.length + 40);
  return (
    <ResponsiveContainer width="100%" height={h}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 48, bottom: 4, left: 8 }}>
        <CartesianGrid horizontal={false} stroke={GRID} strokeDasharray="3 3" />
        <XAxis type="number" domain={domain ?? [0, "auto"]} tick={AXIS} axisLine={false} tickLine={false} tickFormatter={(v) => format(Number(v))} />
        <YAxis type="category" dataKey={labelKey} width={170} tick={AXIS} axisLine={false} tickLine={false} />
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
  return (
    <ResponsiveContainer width="100%" height={Math.max(200, 40 * data.length + 30)}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 40, bottom: 4, left: 8 }}>
        <CartesianGrid horizontal={false} stroke={GRID} strokeDasharray="3 3" />
        <XAxis type="number" domain={[-max, max]} ticks={[-max, -max / 2, 0, max / 2, max]} tick={AXIS} axisLine={false} tickLine={false} tickFormatter={(v) => Number(v).toFixed(1)} />
        <YAxis type="category" dataKey="feature" width={180} tick={AXIS} axisLine={false} tickLine={false} />
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
  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
        <CartesianGrid vertical={false} stroke={GRID} strokeDasharray="3 3" />
        <XAxis dataKey="group" tick={AXIS} axisLine={false} tickLine={false} interval={0} />
        <YAxis tick={AXIS} axisLine={false} tickLine={false} domain={[0, 1]} />
        <Tooltip cursor={{ fill: "var(--muted)", opacity: 0.4 }} contentStyle={tooltipStyle} formatter={(v) => format(Number(v))} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {metrics.map((m, i) => (
          <Bar key={m.key} dataKey={m.key} name={m.label} fill={PALETTE[i % PALETTE.length]} radius={[3, 3, 0, 0]} isAnimationActive={false} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
