import type { Metadata } from "next";
import { api, fmt, titleCase, type FairnessGroup, type Mitigated, type TaskName } from "@/lib/api";
import { GroupedBars } from "@/components/charts/bar-charts";
import { OfflineNotice } from "@/components/offline-notice";
import { TaskPicker } from "@/components/task-picker";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

export const metadata: Metadata = { title: "Fairness" };

const TASKS: TaskName[] = ["at_risk", "math_score", "performance_level"];
const BINARY_METRICS = [
  { key: "selection_rate", label: "selection rate" },
  { key: "tpr", label: "recall (TPR)" },
  { key: "fpr", label: "FPR" },
  { key: "precision", label: "precision" },
];

export default async function FairnessPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const raw = (await searchParams).task;
  const v = Array.isArray(raw) ? raw[0] : raw;
  const task: TaskName = TASKS.includes(v as TaskName) ? (v as TaskName) : "at_risk";
  const [info, fair] = await Promise.all([api.model(task), api.fairness(task)]);
  if (!info || !fair) return <OfflineNotice />;

  const attributes = Array.from(new Set(fair.groups.map((g) => g.attribute)));
  const metrics =
    info.kind === "binary"
      ? BINARY_METRICS
      : info.kind === "regression"
        ? [{ key: "mae", label: "MAE" }]
        : [{ key: "accuracy", label: "accuracy" }];

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-6">
      <section className="flex flex-col gap-3">
        <h1 className="font-display text-3xl font-medium tracking-tight sm:text-4xl">Fairness audit</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Hold-out performance sliced by sensitive attribute. For an early-warning tool the gap that matters most is
          recall: are at-risk students caught at the same rate in every group? Some difference in selection rate is
          expected because base rates genuinely differ, for example free/reduced-lunch students are at risk about
          twice as often. A disparate-impact ratio below 0.8 is the conventional four-fifths warning level.
        </p>
        <TaskPicker tasks={TASKS} current={task} />
      </section>

      {fair.mitigated && <MitigationCard m={fair.mitigated} before={fair.groups} threshold={info.threshold} />}

      {attributes.map((attr) => {
        const groups = fair.groups.filter((g) => g.attribute === attr);
        const gaps = fair.summary[attr] ?? {};
        return (
          <Card key={attr} className="glass">
            <CardHeader>
              <CardTitle>{titleCase(attr)}</CardTitle>
              <CardDescription className="flex flex-wrap gap-2">
                {Object.entries(gaps).map(([k, val]) => (
                  <Badge key={k} variant={k === "disparate_impact_ratio" && val < 0.8 ? "destructive" : "outline"}>
                    {k.replace(/_/g, " ")}: {fmt(val, 3)}
                  </Badge>
                ))}
              </CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-1 gap-6 xl:grid-cols-5">
              <div className="xl:col-span-3">
                <GroupedBars
                  data={groups.map((g) => ({ group: g.group, ...pick(g, metrics.map((m) => m.key)) }))}
                  metrics={metrics}
                  format="fixed2"
                />
              </div>
              <div className="overflow-x-auto xl:col-span-2">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Group</TableHead>
                      <TableHead className="text-right">n</TableHead>
                      {metrics.map((m) => (
                        <TableHead key={m.key} className="text-right">
                          {m.label}
                        </TableHead>
                      ))}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {groups.map((g) => (
                      <TableRow key={g.group}>
                        <TableCell>{g.group}</TableCell>
                        <TableCell className="text-right font-mono text-xs">{g.n}</TableCell>
                        {metrics.map((m) => (
                          <TableCell key={m.key} className="text-right font-mono text-xs">
                            {fmt(g[m.key as keyof FairnessGroup] as number, 2)}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}

function MitigationCard({ m, before, threshold }: { m: Mitigated; before: FairnessGroup[]; threshold: number | null }) {
  const b = new Map(before.filter((g) => g.attribute === m.attribute).map((g) => [g.group, g]));
  const rows = m.groups
    .filter((g) => g.attribute === m.attribute)
    .map((g) => ({ group: g.group, before: b.get(g.group), after: g, threshold: m.thresholds[g.group] }));
  const gapBefore = Object.fromEntries(before.filter((g) => g.attribute === m.attribute).map((g) => [g.group, g.tpr ?? 0]));
  const tprGapBefore = Math.max(...Object.values(gapBefore)) - Math.min(...Object.values(gapBefore));
  const tprGapAfter = m.summary[m.attribute]?.tpr_gap ?? 0;
  const chart = rows.flatMap((r) => [
    { group: `${r.group} (global)`, recall: r.before?.tpr ?? 0, "selection rate": r.before?.selection_rate ?? 0 },
    { group: `${r.group} (per-group)`, recall: r.after.tpr ?? 0, "selection rate": r.after.selection_rate ?? 0 },
  ]);
  return (
    <Card className="glass border-primary/40">
      <CardHeader>
        <CardTitle>Mitigation: recall equalised across {titleCase(m.attribute)}</CardTitle>
        <CardDescription className="flex flex-wrap gap-2">
          <Badge variant="outline">recall gap before: {fmt(tprGapBefore, 3)}</Badge>
          <Badge variant="outline">recall gap after: {fmt(tprGapAfter, 3)}</Badge>
          <Badge variant="outline">
            overall recall {fmt(m.overall_before.recall, 2)} to {fmt(m.overall_after.recall, 2)}
          </Badge>
          <Badge variant="outline">
            precision {fmt(m.overall_before.precision, 2)} to {fmt(m.overall_after.precision, 2)}
          </Badge>
          <Badge variant="outline">
            flagged {fmt(m.overall_before.flagged_rate * 100, 1)}% to {fmt(m.overall_after.flagged_rate * 100, 1)}%
          </Badge>
        </CardDescription>
      </CardHeader>
      <CardContent className="grid grid-cols-1 gap-6 xl:grid-cols-5">
        <div className="xl:col-span-3">
          <GroupedBars data={chart} metrics={[{ key: "recall", label: "recall (TPR)" }, { key: "selection rate", label: "selection rate" }]} format="fixed2" />
        </div>
        <div className="flex flex-col gap-3 xl:col-span-2">
          <p className="text-sm text-muted-foreground">
            One global threshold over-flags one group and under-serves the other. Choosing the cut-off per group on
            out-of-fold probabilities gives every group the same target recall. The model is unchanged; the API
            returns both flags so the choice stays visible.
          </p>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Group</TableHead>
                  <TableHead className="text-right">threshold</TableHead>
                  <TableHead className="text-right">recall</TableHead>
                  <TableHead className="text-right">selection</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((r) => (
                  <TableRow key={r.group}>
                    <TableCell>{r.group}</TableCell>
                    <TableCell className="text-right font-mono text-xs">
                      {fmt(threshold ?? 0, 2)} → {fmt(r.threshold, 2)}
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs">
                      {fmt(r.before?.tpr ?? 0, 2)} → {fmt(r.after.tpr ?? 0, 2)}
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs">
                      {fmt(r.before?.selection_rate ?? 0, 2)} → {fmt(r.after.selection_rate ?? 0, 2)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function pick(g: FairnessGroup, keys: string[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const k of keys) {
    const val = g[k as keyof FairnessGroup];
    if (typeof val === "number") out[k] = val;
  }
  return out;
}
