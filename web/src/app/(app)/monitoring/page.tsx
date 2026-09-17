import type { Metadata } from "next";
import { Activity, AlertTriangle, CheckCircle2, Database, Radar, Sigma } from "lucide-react";
import { api, fmt, titleCase, type DriftFeature, type DriftReport, type TaskName } from "@/lib/api";
import { RankedBars } from "@/components/charts/bar-charts";
import { OfflineNotice } from "@/components/offline-notice";
import { TaskPicker } from "@/components/task-picker";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Accent, Chip, PageHero, Panel, SectionHeading, Tile, type Tone } from "@/components/splash";

export const metadata: Metadata = { title: "Monitoring" };

const TASKS: TaskName[] = ["at_risk", "math_score", "performance_level"];
const STATUS_VARIANT = { ok: "secondary", warn: "outline", alert: "destructive", insufficient: "outline" } as const;
const STATUS_TEXT = {
  ok: "stable",
  warn: "worth a look",
  alert: "material shift",
  insufficient: "waiting for traffic",
} as const;
const STATUS_TONE: Record<DriftReport["status"], Tone> = { ok: "green", warn: "amber", alert: "rose", insufficient: "blue" };
const STATUS_CHIP = { ok: "default", warn: "warn", alert: "alert", insufficient: "default" } as const;

export default async function MonitoringPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const raw = (await searchParams).task;
  const v = Array.isArray(raw) ? raw[0] : raw;
  const task: TaskName = TASKS.includes(v as TaskName) ? (v as TaskName) : "at_risk";
  const [info, drift] = await Promise.all([api.model(task), api.drift(task)]);
  if (!info || !drift) return <OfflineNotice />;

  const waiting = drift.status === "insufficient";

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-8">
      <PageHero
        compact
        eyebrows={[`${drift.n_current} scored in the window`, `model v${drift.model_version}`]}
        title="Drift monitoring"
        description="The API keeps the last few thousand scored students in memory and compares them with the training population. Population stability index (PSI) per input flags a shift in who is being scored; a Kolmogorov-Smirnov test on the model output flags a shift in what the model says. PSI below 0.10 is stable, 0.10 to 0.25 is worth a look, above 0.25 is a material shift."
      >
        <TaskPicker tasks={TASKS} current={task} />
      </PageHero>

      <div className="flex justify-center">
        <Chip icon={drift.status === "ok" ? CheckCircle2 : AlertTriangle} tone={STATUS_CHIP[drift.status]}>
          {titleCase(task)}: <span className="font-medium">{STATUS_TEXT[drift.status]}</span>
          <span className="text-xs text-muted-foreground">
            {" "}
            computed {new Date(drift.computed_at).toLocaleString()}
          </span>
        </Chip>
      </div>

      <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4" aria-label="Window">
        <Tile tone={STATUS_TONE[drift.status]} icon={Radar} label="Status" value={STATUS_TEXT[drift.status]} hint={`${drift.features.length} inputs checked`} />
        <Tile tone="violet" icon={Database} label="Scored students" value={String(drift.n_current)} hint={`against ${drift.n_reference} training students`} />
        <Tile
          tone="blue"
          icon={Sigma}
          label="Output PSI"
          value={drift.scores ? fmt(drift.scores.psi, 3) : "n/a"}
          hint={drift.scores ? `KS ${fmt(drift.scores.ks_statistic, 3)}, p = ${drift.scores.ks_pvalue.toExponential(2)}` : "needs more traffic"}
        />
        <Tile
          tone="amber"
          icon={Activity}
          label="Mean output"
          value={drift.scores ? fmt(drift.scores.current_mean, 3) : "n/a"}
          hint={drift.scores ? `${fmt(drift.scores.reference_mean, 3)} in training` : `needs ${drift.min_rows} rows`}
        />
      </section>

      {waiting ? (
        <Panel tone="blue" title="Waiting for traffic" description={`At least ${drift.min_rows} predictions are needed before a comparison means anything.`}>
          <EmptyState drift={drift} />
        </Panel>
      ) : (
        <section className="flex flex-col gap-6" aria-labelledby="drift-heading">
          <SectionHeading
            id="drift-heading"
            eyebrow="Population stability"
            title={
              <>
                Who is scored, <Accent>what the model says</Accent>
              </>
            }
            description="PSI per input and on the model output. The dashed line is the 0.25 alert level."
          />
          <Report drift={drift} />
        </section>
      )}
    </div>
  );
}

function EmptyState({ drift }: { drift: DriftReport }) {
  return (
    <div className="flex flex-col gap-3 text-sm text-muted-foreground">
      <p>
        The window holds {drift.n_current}. Score some students on the Predict page, send a batch to{" "}
        <code>/predict/{"{task}"}/batch</code>, or post a CSV worth of rows to <code>/monitoring/drift/{"{task}"}</code> to
        compare a cohort directly.
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
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
      <Panel tone="violet" className="xl:col-span-3" title="PSI per input" description="Colour is the status of each input.">
        <RankedBars
          data={bars}
          valueKey="value"
          colorKey="status"
          colorMap={{ ok: "var(--chart-1)", warn: "var(--risk-moderate)", alert: "var(--risk-critical)" }}
          format="fixed3"
          domain={[0, maxPsi]}
          reference={{ value: 0.25, label: "alert" }}
        />
      </Panel>
      <Panel tone="blue" className="xl:col-span-2" title="Largest shifts" description="The bin that moved most for each input." contentClassName="overflow-x-auto">
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
      </Panel>
    </div>
  );
}

function describeShift(f: DriftFeature): string {
  const top = f.top_shifts[0];
  if (!top) return "";
  return `${top.bin}: ${(top.reference * 100).toFixed(0)}% to ${(top.current * 100).toFixed(0)}%`;
}
