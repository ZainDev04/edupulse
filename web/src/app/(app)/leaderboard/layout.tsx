import { TaskPicker } from "@/components/task-picker";
import { PageHero } from "@/components/splash";
import { TASKS } from "@/lib/tasks";

/** The hero stays mounted while the task changes, so only the content below it reloads. */
export default function LeaderboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-8">
      <PageHero
        compact
        eyebrows={["Ten model families", "Repeated stratified cross-validation, 800 students"]}
        title="Model leaderboard"
        description="Every candidate scored with repeated stratified cross-validation on the 800 training students. The best non-baseline family is then tuned with Optuna and fitted once on the full training split."
      >
        <TaskPicker tasks={TASKS} />
      </PageHero>
      {children}
    </div>
  );
}
