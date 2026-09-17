import { TaskPicker } from "@/components/task-picker";
import { PageHero } from "@/components/splash";
import { TASKS } from "@/lib/tasks";

/** The hero stays mounted while the task changes, so only the content below it reloads. */
export default function ExplainLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-8">
      <PageHero
        compact
        eyebrows={["SHAP and permutation importance", "300 hold-out students"]}
        title="Explainability"
        description="Two independent views of what the model relies on. SHAP values are computed on the encoded features and summed back to the original columns, so one-hot categories appear as a single bar. Permutation importance measures how much the hold-out metric drops when a column is shuffled."
      >
        <TaskPicker tasks={TASKS} />
      </PageHero>
      {children}
    </div>
  );
}
