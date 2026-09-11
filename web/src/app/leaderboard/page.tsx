import type { Metadata } from "next";
import { api, fmt, titleCase, type LeaderboardRow, type TaskName } from "@/lib/api";
import { RankedBars } from "@/components/charts/bar-charts";
import { OfflineNotice } from "@/components/offline-notice";
import { StatCard } from "@/components/stat-card";
import { TaskPicker } from "@/components/task-picker";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Award, Clock3, Cpu, Target } from "lucide-react";

export const metadata: Metadata = { title: "Leaderboard" };

const TASKS: TaskName[] = ["at_risk", "math_score", "performance_level"];

function pickTask(raw: string | string[] | undefined): TaskName {
  const v = Array.isArray(raw) ? raw[0] : raw;
  return TASKS.includes(v as TaskName) ? (v as TaskName) : "at_risk";
}

export default async function LeaderboardPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const task = pickTask((await searchParams).task);
  const [info, board] = await Promise.all([api.model(task), api.leaderboard(task)]);
  if (!info || !board) return <OfflineNotice />;

  const metric = info.primary_metric;
  const meanKey = `${metric}_mean`;
  const stdKey = `${metric}_std`;
  const rows = board.rows as LeaderboardRow[];
  const chart = rows.map((r) => ({
    name: String(r.model),
    value: Number(r[meanKey]),
    std: Number(r[stdKey]),
    family: String(r.family),
  }));
  const best = rows[0];
  const min = Math.min(...chart.map((c) => c.value));
  const secondary = Object.keys(rows[0]).filter((k) => k.endsWith("_mean") && k !== meanKey);

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-6">
      <section className="flex flex-col gap-3">
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Model leaderboard</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Every candidate scored with repeated stratified cross-validation on the 800 training students. The best
          non-baseline family is then tuned with Optuna and fitted once on the full training split.
        </p>
        <TaskPicker tasks={TASKS} current={task} />
      </section>

      {info.leakage_note && (
        <Card className="glass border-risk-moderate/40">
          <CardContent className="text-sm">
            <span className="font-medium">Leakage note. </span>
            {info.leakage_note}
          </CardContent>
        </Card>
      )}

      <section className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Selected model" value={titleCase(info.model)} hint={info.model_class} icon={Award} />
        <StatCard label={`CV ${metric}`} value={fmt(info.cv_score)} hint={`± ${fmt(Number(best[stdKey]))} across folds`} icon={Target} />
        <StatCard
          label={info.kind === "regression" ? "Hold-out R²" : info.kind === "binary" ? "Hold-out ROC-AUC" : "Hold-out accuracy"}
          value={fmt(info.kind === "regression" ? info.test_metrics.r2 : info.kind === "binary" ? info.test_metrics.roc_auc : info.test_metrics.accuracy)}
          hint="200 held-out students"
          icon={Cpu}
        />
        <StatCard label="Fit time" value={`${Number(best.fit_seconds).toFixed(2)}s`} hint="mean per fold, selected model" icon={Clock3} />
      </section>

      <Card className="glass">
        <CardHeader>
          <CardTitle>Cross-validated {metric}</CardTitle>
          <CardDescription>Mean with standard deviation across 10 folds. Colour is the model family.</CardDescription>
        </CardHeader>
        <CardContent>
          <RankedBars
            data={chart}
            valueKey="value"
            errorKey="std"
            colorKey="family"
            domain={[Math.max(0, Math.floor(min * 10) / 10 - 0.1), 1]}
          />
          <div className="mt-2 flex flex-wrap gap-2">
            {Array.from(new Set(chart.map((c) => c.family))).map((f) => (
              <Badge key={f} variant="outline">
                {f}
              </Badge>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card className="glass">
        <CardHeader>
          <CardTitle>All metrics</CardTitle>
          <CardDescription>Cross-validation means for every metric the task records.</CardDescription>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>#</TableHead>
                <TableHead>Model</TableHead>
                <TableHead>Family</TableHead>
                <TableHead className="text-right">{metric}</TableHead>
                {secondary.map((k) => (
                  <TableHead key={k} className="text-right">
                    {k.replace("_mean", "").replace("neg_", "")}
                  </TableHead>
                ))}
                <TableHead className="text-right">fit (s)</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((r) => (
                <TableRow key={String(r.model)}>
                  <TableCell className="font-mono text-xs">{r.rank}</TableCell>
                  <TableCell className="font-medium">{String(r.model)}</TableCell>
                  <TableCell className="text-muted-foreground">{String(r.family)}</TableCell>
                  <TableCell className="text-right font-mono text-xs">{fmt(Number(r[meanKey]))}</TableCell>
                  {secondary.map((k) => (
                    <TableCell key={k} className="text-right font-mono text-xs">
                      {fmt(Math.abs(Number(r[k])))}
                    </TableCell>
                  ))}
                  <TableCell className="text-right font-mono text-xs">{Number(r.fit_seconds).toFixed(2)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
