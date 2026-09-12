import Link from "next/link";
import { ArrowRight, Gauge, GraduationCap, ShieldAlert, Users } from "lucide-react";
import { api, fmt, pct, titleCase, type TaskName, TASK_LABEL } from "@/lib/api";
import { StatCard } from "@/components/stat-card";
import { RankedBars } from "@/components/charts/bar-charts";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { OfflineNotice } from "@/components/offline-notice";

function headline(kind: string, m: Record<string, number>): string {
  if (kind === "binary") return `ROC-AUC ${fmt(m.roc_auc)}, recall ${pct(m.recall, 0)}`;
  if (kind === "regression") return `R² ${fmt(m.r2)}, RMSE ${fmt(m.rmse, 2)}`;
  return `accuracy ${pct(m.accuracy)}, macro-F1 ${fmt(m.f1_macro)}`;
}

export default async function OverviewPage() {
  const [stats, models] = await Promise.all([api.stats(), api.models()]);
  if (!stats || !models) return <OfflineNotice />;

  const atRisk = models.at_risk;
  const lunch = stats.by_attribute.lunch ?? [];
  const prep = stats.by_attribute.test_preparation_course ?? [];
  const parental = [...(stats.by_attribute.parental_level_of_education ?? [])].sort(
    (a, b) => b.at_risk_rate - a.at_risk_rate,
  );

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-6">
      <section className="flex flex-col gap-3">
        <h1 className="font-display text-3xl font-medium tracking-tight sm:text-4xl">Student performance intelligence</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Three registered models trained on {stats.n_students.toLocaleString()} students. The at-risk model
          flags students likely to average below 60 from background information alone, before any exam is taken.
          Every prediction can be explained with SHAP and every model ships with a fairness audit.
        </p>
        <div className="flex flex-wrap gap-2">
          <Button render={<Link href="/predict" />} nativeButton={false} size="sm">
            Score a student <ArrowRight className="size-4" aria-hidden="true" />
          </Button>
          <Button render={<Link href="/leaderboard" />} nativeButton={false} size="sm" variant="outline">
            Model leaderboard
          </Button>
        </div>
      </section>

      <section className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4" aria-label="Key figures">
        <StatCard label="Students" value={stats.n_students.toLocaleString()} hint="rows after cleaning" icon={Users} />
        <StatCard label="At-risk rate" value={pct(stats.at_risk_rate)} hint="average score below 60" icon={ShieldAlert} />
        <StatCard label="Average score" value={stats.average_score.toFixed(1)} hint="mean of math, reading, writing" icon={GraduationCap} />
        <StatCard
          label="At-risk model ROC-AUC"
          value={atRisk ? fmt(atRisk.test_metrics.roc_auc) : "n/a"}
          hint={atRisk ? `${titleCase(atRisk.model)}, hold-out` : "not trained"}
          icon={Gauge}
        />
      </section>

      <section className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card className="glass lg:col-span-2">
          <CardHeader>
            <CardTitle>Registered models</CardTitle>
            <CardDescription>Latest version per task, hold-out metrics on 200 held-out students.</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Task</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Cross-validation</TableHead>
                  <TableHead>Hold-out</TableHead>
                  <TableHead>Version</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {(Object.keys(models) as TaskName[]).map((t) => {
                  const m = models[t];
                  return (
                    <TableRow key={t}>
                      <TableCell className="font-medium">
                        <Link href={`/leaderboard?task=${t}`} className="hover:underline">
                          {TASK_LABEL[t]}
                        </Link>
                        {m.leakage_note && (
                          <Badge variant="outline" className="ml-2 text-[10px]">
                            leakage case study
                          </Badge>
                        )}
                      </TableCell>
                      <TableCell>{titleCase(m.model)}</TableCell>
                      <TableCell className="font-mono text-xs">
                        {m.cv_metric} {fmt(m.cv_score)}
                      </TableCell>
                      <TableCell className="font-mono text-xs">{headline(m.kind, m.test_metrics)}</TableCell>
                      <TableCell
                        className="whitespace-nowrap font-mono text-xs text-muted-foreground"
                        title={m.mlflow_run_id ? `MLflow run ${m.mlflow_run_id}` : undefined}
                      >
                        {m.version}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        <Card className="glass">
          <CardHeader>
            <CardTitle>What predicts risk</CardTitle>
            <CardDescription>At-risk rate by group in the training data.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4 text-sm">
            <GroupRow title="Lunch" rows={lunch} />
            <GroupRow title="Test preparation" rows={prep} />
          </CardContent>
        </Card>
      </section>

      <Card className="glass">
        <CardHeader>
          <CardTitle>At-risk rate by parental education</CardTitle>
          <CardDescription>
            Dashed line is the overall rate ({pct(stats.at_risk_rate)}). Higher parental education tracks lower risk.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <RankedBars
            data={parental.map((g) => ({ name: g.group, rate: g.at_risk_rate, n: g.n }))}
            valueKey="rate"
            format="pct"
            domain={[0, 0.6]}
            reference={{ value: stats.at_risk_rate, label: "overall" }}
          />
        </CardContent>
      </Card>
    </div>
  );
}

function GroupRow({ title, rows }: { title: string; rows: { group: string; n: number; at_risk_rate: number }[] }) {
  return (
    <div>
      <div className="mb-1 text-xs font-medium tracking-wide text-muted-foreground uppercase">{title}</div>
      <ul className="flex flex-col gap-1">
        {rows.map((r) => (
          <li key={r.group} className="flex items-center justify-between gap-3">
            <span>{r.group}</span>
            <span className="flex items-center gap-2 font-mono text-xs">
              <span className="text-muted-foreground">n={r.n}</span>
              <span className="w-12 text-right">{pct(r.at_risk_rate, 0)}</span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
