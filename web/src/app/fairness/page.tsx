import type { Metadata } from "next";
import { api, fmt, titleCase, type FairnessGroup, type TaskName } from "@/lib/api";
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
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Fairness audit</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Hold-out performance sliced by sensitive attribute. For an early-warning tool the gap that matters most is
          recall: are at-risk students caught at the same rate in every group? Some difference in selection rate is
          expected because base rates genuinely differ, for example free/reduced-lunch students are at risk about
          twice as often. A disparate-impact ratio below 0.8 is the conventional four-fifths warning level.
        </p>
        <TaskPicker tasks={TASKS} current={task} />
      </section>

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

function pick(g: FairnessGroup, keys: string[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const k of keys) {
    const val = g[k as keyof FairnessGroup];
    if (typeof val === "number") out[k] = val;
  }
  return out;
}
