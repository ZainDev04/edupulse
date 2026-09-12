/**
 * Typed client for the EduPulse FastAPI backend.
 *
 * Server components call the backend directly (API_URL, server-only).
 * Client components go through the Next.js proxy at /api/* so the browser
 * never needs to know where the backend lives and CORS is a non-issue.
 */

export const API_URL = process.env.API_URL ?? "http://localhost:8000";

export type TaskName = "at_risk" | "math_score" | "performance_level";

export const TASK_LABEL: Record<TaskName, string> = {
  at_risk: "At-risk early warning",
  math_score: "Math score prediction",
  performance_level: "Performance level",
};

export const TASK_SLUG: Record<TaskName, string> = {
  at_risk: "at-risk",
  math_score: "math-score",
  performance_level: "performance-level",
};

export interface ModelInfo {
  task: TaskName;
  kind: "binary" | "multiclass" | "regression";
  description: string;
  version: string;
  model: string;
  model_class: string;
  features: string[];
  threshold: number | null;
  cv_metric: string;
  cv_score: number;
  primary_metric: string;
  test_metrics: Record<string, number>;
  created_at: string;
  leakage_note: string | null;
  mlflow_run_id: string | null;
}

export interface Health {
  status: string;
  version: string;
  models: Record<TaskName, string | null>;
}

export interface LeaderboardRow {
  rank: number;
  model: string;
  family: string;
  fit_seconds: number;
  [metric: string]: number | string;
}

export interface FairnessGroup {
  attribute: string;
  group: string;
  n: number;
  prevalence?: number;
  selection_rate?: number;
  tpr?: number;
  fpr?: number;
  precision?: number;
  roc_auc?: number | null;
  mae?: number;
  mean_residual?: number;
  accuracy?: number;
}

export interface Fairness {
  groups: FairnessGroup[];
  summary: Record<string, Record<string, number>>;
}

export interface Importance {
  shap: { feature: string; mean_abs_shap: number }[];
  permutation: { feature: string; importance_mean: number; importance_std: number }[];
  shap_kind: string | null;
}

export interface GroupStat {
  group: string;
  n: number;
  at_risk_rate: number;
  avg_score: number;
}

export interface DatasetStats {
  n_students: number;
  at_risk_rate: number;
  average_score: number;
  score_means: Record<string, number>;
  performance_level_share: Record<string, number>;
  by_attribute: Record<string, GroupStat[]>;
}

export interface Contribution {
  feature: string;
  value: string | number | null;
  contribution: number;
}

export interface AtRiskPrediction {
  probability: number;
  at_risk: boolean;
  threshold: number;
  risk_band: "low" | "moderate" | "high" | "critical";
  model_version: string;
  explanation?: Contribution[] | null;
}

export interface ScorePrediction {
  prediction: number;
  model_version: string;
  explanation?: Contribution[] | null;
}

export interface LevelPrediction {
  label: "low" | "medium" | "high";
  probabilities: Record<string, number>;
  model_version: string;
  explanation?: Contribution[] | null;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new ApiError(res.status, `${path} returned ${res.status}`);
  }
  return res.json() as Promise<T>;
}

/** Returns null instead of throwing so pages can render an "API offline" state. */
async function tryGet<T>(path: string): Promise<T | null> {
  try {
    return await get<T>(path);
  } catch {
    return null;
  }
}

export const api = {
  health: () => tryGet<Health>("/health"),
  models: () => tryGet<Record<TaskName, ModelInfo>>("/models"),
  model: (task: TaskName) => tryGet<ModelInfo>(`/models/${TASK_SLUG[task]}`),
  leaderboard: (task: TaskName) =>
    tryGet<{ task: TaskName; rows: LeaderboardRow[] }>(`/models/${TASK_SLUG[task]}/leaderboard`),
  fairness: (task: TaskName) => tryGet<Fairness>(`/models/${TASK_SLUG[task]}/fairness`),
  importance: (task: TaskName) => tryGet<Importance>(`/models/${TASK_SLUG[task]}/importance`),
  stats: () => tryGet<DatasetStats>("/stats"),
};

export const GENDERS = ["female", "male"] as const;
export const ETHNICITIES = ["group A", "group B", "group C", "group D", "group E"] as const;
export const PARENTAL_EDUCATION = [
  "some high school",
  "high school",
  "some college",
  "associate's degree",
  "bachelor's degree",
  "master's degree",
] as const;
export const LUNCH = ["free/reduced", "standard"] as const;
export const TEST_PREP = ["none", "completed"] as const;

export function fmt(n: number | null | undefined, digits = 3): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "n/a";
  return n.toFixed(digits);
}

export function pct(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "n/a";
  return `${(n * 100).toFixed(digits)}%`;
}

export function titleCase(s: string): string {
  return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
