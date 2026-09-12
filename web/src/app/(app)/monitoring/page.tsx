import type { Metadata } from "next";
import { api, fmt, titleCase, type DriftFeature, type DriftReport, type TaskName } from "@/lib/api";
import { RankedBars } from "@/components/charts/bar-charts";
import { OfflineNotice } from "@/components/offline-notice";
import { TaskPicker } from "@/components/task-picker";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

export const metadata: Metadata = { title: "Monitoring" };

const TASKS: TaskName[] = ["at_risk", "math_score", "performance_level"];
const STATUS_VARIANT = { ok: "secondary", warn: "outline", alert: "destructive", insufficient: "outline" } as const;
const STATUS_TEXT = {
  ok: "stable",
  warn: "worth a look",
  alert: "material shift",
  insufficient: "waiting for traffic",
} as const;

export default async function MonitoringPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const raw = (await searchParams).task;
  const v = Array.isArray(raw) ? raw[0] : raw;
  const task: TaskName = TASKS.includes(v as TaskName) ? (v as TaskName) : "at_risk";
  const [info, drift] = await Promise.all([api.model(task), api.drift(task)]);
  if (!info || !drift) return <OfflineNotice />;

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-6">
      <section className="flex flex-col gap-3">
        <h1 className="font-display text-3xl font-medium tracking-tight sm:text-4xl">Drift monitoring</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          The API keeps the last few thousand scored students in memory and compares them with the training
          population. Population stability index (PSI) per input flags a shift in who is being scored; a
          Kolmogorov-Smirnov test on the model output flags a shift in what the model says. PSI below 0.10 is
          stable, 0.10 to 0.25 is worth a look, above 0.25 is a material shift.
        </p>
        <TaskPicker tasks={TASKS} current={task} />
      </section>

      <Card className="glass">
        <CardHeader>
          <CardTitle className="flex flex-wrap items-center gap-2">
            {titleCase(task)}
            <Badge variant={STATUS_VARIANT[drift.status]}>{STATUS_TEXT[drift.status]}</Badge>
          </CardTitle>
          <CardDescription>
            {drift.n_current} scored students in the window against {drift.n_reference} training students, model v
            {drift.model_version}, computed {new Date(drift.computed_at).toLocaleString()}.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {drift.status === "insufficient" ? (
            <EmptyState drift={drift} />
          ) : (
            <Report drift={drift} />
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function EmptyState({ drift }: { drift: DriftReport }) {
  return (
    <div className="flex flex-col gap-3 text-sm text-muted-foreground">
      <p>
        At least {drift.min_rows} predictions are needed before a comparison means anything; the window holds{" "}
        {drift.n_current}. Score some students on the Predict page, send a batch to <code>/predict/{"{task}"}/batch</code>,
        or post a CSV worth of rows to <code>/monitoring/drift/{"{task}"}</code> to compare a cohort directly.
      </p>
      <p>
        From a terminal: <code>edupulse drift --task at_risk --path new_intake.csv</code>.
      </p>
    </div>
  );
}

function Report({ drift }: { drift: DriftReport }) {
  const bars = drift.features.map((f) => ({ name: f.feature, value: f.psi, status: f.status }));
  if (drift.scores) bars.push({ name: "model output", value: drift.scores.psi, status: drift.scores.status });
  const maxPsi = Math.max(0.3, ...bars.map((b) => b.value));
  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-5">
      <div className="xl:col-span-3">
        <RankedBars
          data={bars}
          valueKey="value"
          colorKey="status"
          colorMap={{ ok: "var(--chart-1)", warn: "var(--risk-moderate)", alert: "var(--risk-critical)" }}
          format="fixed3"
          domain={[0, maxPsi]}
          reference={{ value: 0.25, label: "alert" }}
        />
      </div>
      <div className="flex flex-col gap-4 xl:col-span-2">
        {drift.scores && (
          <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
            <dt className="text-muted-foreground">Output PSI</dt>
            <dd className="font-mono text-xs">{fmt(drift.scores.psi, 3)}</dd>
            <dt className="text-muted-foreground">KS statistic</dt>
            <dd className="font-mono text-xs">
              {fmt(drift.scores.ks_statistic, 3)} (p = {drift.scores.ks_pvalue.toExponential(2)})
            </dd>
            <dt className="text-muted-foreground">Mean output</dt>
            <dd className="font-mono text-xs">
              {fmt(drift.scores.reference_mean, 3)} training, {fmt(drift.scores.current_mean, 3)} now
            </dd>
          </dl>
        )}
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Feature</TableHead>
                <TableHead className="text-right">PSI</TableHead>
                <TableHead>Largest shift</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {drift.features.map((f) => (
                <TableRow key={f.feature}>
                  <TableCell>
                    {f.feature.replace(/_/g, " ")}
                    {f.status !== "ok" && (
                      <Badge variant={STATUS_VARIANT[f.status]} className="ml-2 text-[10px]">
                        {f.status}
                      </Badge>
                    )}
                  </TableCell>
                  <TableCell className="text-right font-mono text-xs">{fmt(f.psi, 3)}</TableCell>
                  <TableCell className="font-mono text-xs text-muted-foreground">{describeShift(f)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </div>
    </div>
  );
}

function describeShift(f: DriftFeature): string {
  const top = f.top_shifts[0];
  if (!top) return "";
  return `${top.bin}: ${(top.reference * 100).toFixed(0)}% to ${(top.current * 100).toFixed(0)}%`;
}
