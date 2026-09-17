"use client";

import { useState, useTransition } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Loader2 } from "lucide-react";
import { cn } from "cn";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TASK_LABEL, type TaskName } from "@/lib/api";

/**
 * Task switcher that writes ?task= to the URL so pages stay server-rendered
 * and deep-linkable. The push runs in a transition, so the current page
 * stays on screen (no loading skeleton) while the next one is fetched; the
 * tapped tab takes the active style at once and shows a spinner meanwhile.
 */
export function TaskPicker({ tasks }: { tasks: TaskName[] }) {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const fromUrl = params.get("task");
  const current: TaskName = tasks.includes(fromUrl as TaskName) ? (fromUrl as TaskName) : tasks[0];
  const [pending, start] = useTransition();
  const [target, setTarget] = useState<TaskName | null>(null);
  const shown = pending && target ? target : current;

  return (
    <Tabs
      value={shown}
      onValueChange={(v) => {
        const next = new URLSearchParams(params.toString());
        next.set("task", String(v));
        setTarget(v as TaskName);
        start(() => router.push(`${pathname}?${next.toString()}`));
      }}
    >
      {/* Lives inside the dark hero, so colours are explicit rather than theme tokens */}
      <TabsList
        aria-label="Task"
        aria-busy={pending || undefined}
        className={cn(
          "h-auto flex-wrap justify-center gap-1 rounded-full bg-white/10 p-1 backdrop-blur transition-opacity group-data-horizontal/tabs:h-auto",
          pending && "opacity-80",
        )}
      >
        {tasks.map((t) => (
          <TabsTrigger
            key={t}
            value={t}
            className="h-9 flex-none rounded-full px-4 text-white/75 hover:text-white focus-visible:border-transparent focus-visible:outline-none focus-visible:ring-white/50 data-active:bg-white data-active:text-slate-900 data-active:shadow-none dark:text-white/75 dark:hover:text-white dark:data-active:border-transparent dark:data-active:bg-white dark:data-active:text-slate-900"
          >
            {TASK_LABEL[t]}
            {pending && t === shown && <Loader2 className="size-3.5 animate-spin" aria-hidden="true" />}
          </TabsTrigger>
        ))}
      </TabsList>
    </Tabs>
  );
}
