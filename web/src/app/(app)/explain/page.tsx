import type { Metadata } from "next";
import { api, type TaskName } from "@/lib/api";
import { RankedBars } from "@/components/charts/bar-charts";
import { OfflineNotice } from "@/components/offline-notice";
import { TaskPicker } from "@/components/task-picker";
import { Accent, PageHero, Panel, SectionHeading } from "@/components/splash";

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
  const top = shap[0]?.name.replace(/_/g, " ");

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-8">
      <PageHero
        compact
        eyebrows={[
          imp.shap_kind ? `${imp.shap_kind} SHAP explainer` : "SHAP explainer",
          top ? `Strongest signal: ${top}` : `${info.model_class}`,
        ]}
        title="Explainability"
        description={`Two independent views of what the model relies on. SHAP values are computed on the encoded features and summed back to the original columns, so one-hot categories appear as a single bar. Permutation importance measures how much the hold-out ${permMetric} drops when a column is shuffled.`}
      >
        <TaskPicker tasks={TASKS} current={task} />
      </PageHero>

      <section className="flex flex-col gap-6" aria-labelledby="views-heading">
        <SectionHeading
          id="views-heading"
          eyebrow="Feature importance"
          title={
            <>
              Two views, <Accent>one story</Accent>
            </>
          }
          description="SHAP says how much a feature moves individual predictions. Permutation says how much the model needs it on unseen data."
        />
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
          <Panel
            tone="violet"
            title="Mean absolute SHAP"
            description={`${imp.shap_kind ? `${imp.shap_kind} explainer` : "explainer"}, 300 hold-out students, ${info.model_class}.`}
          >
            {shap.length ? (
              <RankedBars data={shap} valueKey="value" format="fixed3" />
            ) : (
              <p className="text-sm text-muted-foreground">No SHAP values recorded for this model.</p>
            )}
          </Panel>

          <Panel
            tone="blue"
            title="Permutation importance"
            description={`Drop in hold-out ${permMetric}, mean and standard deviation over 10 shuffles.`}
          >
            <RankedBars data={perm} valueKey="value" errorKey="std" format="fixed3" />
          </Panel>
        </div>
      </section>

      <Panel title="How to read this" contentClassName="grid grid-cols-1 gap-4 text-sm text-muted-foreground md:grid-cols-2">
        <p>
          A large SHAP value means the feature moves individual predictions a lot, in either direction. A large
          permutation importance means the model cannot do without the feature on unseen data. Features can rank
          differently on the two: a feature the model uses heavily but that is correlated with another one will have
          high SHAP and low permutation importance.
        </p>
        <p>
          For the at-risk model, lunch type (a proxy for household income) and the test preparation course dominate
          both views. Gender and ethnicity contribute less. Per-student explanations with the same aggregation are
          available on the Predict page.
        </p>
      </Panel>
    </div>
  );
}
