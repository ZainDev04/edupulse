import type { Metadata } from "next";
import { api, type TaskName } from "@/lib/api";
import { RankedBars } from "@/components/charts/bar-charts";
import { OfflineNotice } from "@/components/offline-notice";
import { TaskPicker } from "@/components/task-picker";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export const metadata: Metadata = { title: "Explainability" };

const TASKS: TaskName[] = ["at_risk", "math_score", "performance_level"];

export default async function ExplainPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const raw = (await searchParams).task;
  const v = Array.isArray(raw) ? raw[0] : raw;
  const task: TaskName = TASKS.includes(v as TaskName) ? (v as TaskName) : "at_risk";
  const [info, imp] = await Promise.all([api.model(task), api.importance(task)]);
  if (!info || !imp) return <OfflineNotice />;

  const shap = imp.shap.map((r) => ({ name: r.feature, value: r.mean_abs_shap }));
  const perm = imp.permutation.map((r) => ({ name: r.feature, value: r.importance_mean, std: r.importance_std }));
  const permMetric = info.kind === "binary" ? "ROC-AUC" : info.kind === "regression" ? "R²" : "macro-F1";

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-6">
      <section className="flex flex-col gap-3">
        <h1 className="font-display text-3xl font-medium tracking-tight sm:text-4xl">Explainability</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Two independent views of what the model relies on. SHAP values are computed on the encoded features and
          summed back to the original columns, so one-hot categories appear as a single bar. Permutation importance
          measures how much the hold-out {permMetric} drops when a column is shuffled.
        </p>
        <TaskPicker tasks={TASKS} current={task} />
      </section>

      <section className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card className="glass">
          <CardHeader>
            <CardTitle>Mean absolute SHAP</CardTitle>
            <CardDescription>
              {imp.shap_kind ? `${imp.shap_kind} explainer` : "explainer"}, 300 hold-out students, {info.model_class}.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {shap.length ? (
              <RankedBars data={shap} valueKey="value" format="fixed3" />
            ) : (
              <p className="text-sm text-muted-foreground">No SHAP values recorded for this model.</p>
            )}
          </CardContent>
        </Card>

        <Card className="glass">
          <CardHeader>
            <CardTitle>Permutation importance</CardTitle>
            <CardDescription>Drop in hold-out {permMetric}, mean and standard deviation over 10 shuffles.</CardDescription>
          </CardHeader>
          <CardContent>
            <RankedBars data={perm} valueKey="value" errorKey="std" format="fixed3" />
          </CardContent>
        </Card>
      </section>

      <Card className="glass">
        <CardHeader>
          <CardTitle>How to read this</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-1 gap-4 text-sm text-muted-foreground md:grid-cols-2">
          <p>
            A large SHAP value means the feature moves individual predictions a lot, in either direction. A large
            permutation importance means the model cannot do without the feature on unseen data. Features can rank
            differently on the two: a feature the model uses heavily but that is correlated with another one will
            have high SHAP and low permutation importance.
          </p>
          <p>
            For the at-risk model, lunch type (a proxy for household income) and the test preparation course dominate
            both views. Gender and ethnicity contribute less. Per-student explanations with the same aggregation are
            available on the Predict page.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
