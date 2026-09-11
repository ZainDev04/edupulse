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
      <TabsList aria-label="Task">
        {tasks.map((t) => (
          <TabsTrigger key={t} value={t}>
            {TASK_LABEL[t]}
          </TabsTrigger>
        ))}
      </TabsList>
    </Tabs>
  );
}
