import type { Metadata } from "next";
import { api, type TaskName } from "@/lib/api";
import { OfflineNotice } from "@/components/offline-notice";
import { PredictForm } from "./predict-form";

export const metadata: Metadata = { title: "Predict" };

export default async function PredictPage() {
  const models = await api.models();
  if (!models) return <OfflineNotice />;
  const tasks = Object.keys(models) as TaskName[];
  const thresholds = Object.fromEntries(tasks.map((t) => [t, models[t].threshold])) as Partial<Record<TaskName, number | null>>;

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-6">
      <section className="flex flex-col gap-2">
        <h1 className="font-display text-3xl font-medium tracking-tight sm:text-4xl">Score a student</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Predictions come from the registered pipelines through the REST API. The at-risk task uses background
          only; the other two also take the relevant exam scores.
        </p>
      </section>
      <PredictForm tasks={tasks} thresholds={thresholds} />
    </div>
  );
}
