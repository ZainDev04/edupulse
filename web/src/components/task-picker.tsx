"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TASK_LABEL, type TaskName } from "@/lib/api";

/** Task switcher that writes ?task= to the URL so pages stay server-rendered and deep-linkable. */
export function TaskPicker({ tasks, current }: { tasks: TaskName[]; current: TaskName }) {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  return (
    <Tabs
      value={current}
      onValueChange={(v) => {
        const next = new URLSearchParams(params.toString());
        next.set("task", String(v));
        router.push(`${pathname}?${next.toString()}`);
      }}
    >
      {/* Lives inside the dark hero, so colours are explicit rather than theme tokens */}
      <TabsList aria-label="Task" className="h-auto flex-wrap justify-center gap-1 rounded-full bg-white/10 p-1 backdrop-blur">
        {tasks.map((t) => (
          <TabsTrigger
            key={t}
            value={t}
            className="h-9 flex-none rounded-full px-4 text-white/75 hover:text-white focus-visible:ring-white/50 data-active:bg-white data-active:text-slate-900 data-active:shadow-none dark:text-white/75 dark:hover:text-white dark:data-active:border-transparent dark:data-active:bg-white dark:data-active:text-slate-900"
          >
            {TASK_LABEL[t]}
          </TabsTrigger>
        ))}
      </TabsList>
    </Tabs>
  );
}
