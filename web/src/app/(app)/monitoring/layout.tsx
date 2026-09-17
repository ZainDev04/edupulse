import { TaskPicker } from "@/components/task-picker";
import { PageHero } from "@/components/splash";
import { TASKS } from "@/lib/tasks";

/** The hero stays mounted while the task changes, so only the content below it reloads. */
export default function MonitoringLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-8">
      <PageHero
        compact
        eyebrows={["PSI per input, KS on the output", "Last few thousand scored students"]}
        title="Drift monitoring"
        description="The API keeps the last few thousand scored students in memory and compares them with the training population. Population stability index (PSI) per input flags a shift in who is being scored; a Kolmogorov-Smirnov test on the model output flags a shift in what the model says. PSI below 0.10 is stable, 0.10 to 0.25 is worth a look, above 0.25 is a material shift."
      >
        <TaskPicker tasks={TASKS} />
      </PageHero>
      {children}
    </div>
  );
}
