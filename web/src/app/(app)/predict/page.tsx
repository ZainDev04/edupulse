import type { Metadata } from "next";
import { api, type TaskName } from "@/lib/api";
import { OfflineNotice } from "@/components/offline-notice";
import { PageHero } from "@/components/splash";
import { PredictForm } from "./predict-form";

export const metadata: Metadata = { title: "Predict" };

export default async function PredictPage() {
  const models = await api.models();
  if (!models) return <OfflineNotice />;
  const tasks = Object.keys(models) as TaskName[];
  const thresholds = Object.fromEntries(tasks.map((t) => [t, models[t].threshold])) as Partial<Record<TaskName, number | null>>;

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-8">
      <PageHero
        compact
        eyebrows={["Live scoring", `${tasks.length} pipelines behind one REST API`]}
        title="Score a student"
        description="Predictions come from the registered pipelines through the REST API. The at-risk task uses background only; the other two also take the relevant exam scores."
      />
      <PredictForm tasks={tasks} thresholds={thresholds} />
    </div>
  );
}
