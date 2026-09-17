import type { TaskName } from "@/lib/api";

/** Tasks the task pages can switch between, in picker order. */
export const TASKS: TaskName[] = ["at_risk", "math_score", "performance_level"];

/** Reads ?task= from a page's search params, falling back to the at-risk task. */
export function pickTask(raw: string | string[] | undefined): TaskName {
  const v = Array.isArray(raw) ? raw[0] : raw;
  return TASKS.includes(v as TaskName) ? (v as TaskName) : "at_risk";
}
