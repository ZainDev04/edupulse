import type { Metadata } from "next";
import { api } from "@/lib/api";
import { RankedBars } from "@/components/charts/bar-charts";
import { OfflineNotice } from "@/components/offline-notice";
import { Accent, Panel, SectionHeading } from "@/components/splash";
import { pickTask } from "@/lib/tasks";

export const metadata: Metadata = { title: "Explainability" };

export default async function ExplainPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const task = pickTask((await searchParams).task);
  const [info, imp] = await Promise.all([api.model(task), api.importance(task)]);
  if (!info || !imp) return <OfflineNotice />;

  const shap = imp.shap.map((r) => ({ name: r.feature, value: r.mean_abs_shap }));
  const perm = imp.permutation.map((r) => ({ name: r.feature, value: r.importance_mean, std: r.importance_std }));
  const permMetric = info.kind === "binary" ? "ROC-AUC" : info.kind === "regression" ? "R²" : "macro-F1";

  return (
    <>
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
    </>
  );
}
