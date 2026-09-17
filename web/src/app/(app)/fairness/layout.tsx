import { TaskPicker } from "@/components/task-picker";
import { PageHero } from "@/components/splash";
import { TASKS } from "@/lib/tasks";

/** The hero stays mounted while the task changes, so only the content below it reloads. */
export default function FairnessLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-8">
      <PageHero
        compact
        eyebrows={["Four sensitive attributes", "Recall equalised across lunch groups"]}
        title="Fairness audit"
        description="Hold-out performance sliced by sensitive attribute. For an early-warning tool the gap that matters most is recall: are at-risk students caught at the same rate in every group? Some difference in selection rate is expected because base rates genuinely differ. A disparate-impact ratio below 0.8 is the conventional four-fifths warning level."
      >
        <TaskPicker tasks={TASKS} />
      </PageHero>
      {children}
    </div>
  );
}
