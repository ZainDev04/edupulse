"use client";

import { useState, useTransition } from "react";
import { AlertCircle, Loader2 } from "lucide-react";
import { cn } from "cn";
import {
  ETHNICITIES,
  GENDERS,
  LUNCH,
  PARENTAL_EDUCATION,
  TASK_LABEL,
  TASK_SLUG,
  TEST_PREP,
  fmt,
  pct,
  type AtRiskPrediction,
  type Contribution,
  type LevelPrediction,
  type ScorePrediction,
  type TaskName,
} from "@/lib/api";
import { ContributionBars } from "@/components/charts/bar-charts";
import { ProgressRadial } from "@/components/ui/progress-radial";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

type Form = {
  gender: string;
  race_ethnicity: string;
  parental_level_of_education: string;
  lunch: string;
  test_preparation_course: string;
  math_score: number;
  reading_score: number;
  writing_score: number;
};

const DEFAULTS: Form = {
  gender: "female",
  race_ethnicity: "group C",
  parental_level_of_education: "some college",
  lunch: "standard",
  test_preparation_course: "none",
  math_score: 65,
  reading_score: 70,
  writing_score: 68,
};

const SCORES_FOR: Record<TaskName, (keyof Form)[]> = {
  at_risk: [],
  math_score: ["reading_score", "writing_score"],
  performance_level: ["math_score", "reading_score", "writing_score"],
};

type Result =
  | { task: "at_risk"; data: AtRiskPrediction }
  | { task: "math_score"; data: ScorePrediction }
  | { task: "performance_level"; data: LevelPrediction };

const BAND_CLASS: Record<AtRiskPrediction["risk_band"], string> = {
  low: "text-risk-low",
  moderate: "text-risk-moderate",
  high: "text-risk-high",
  critical: "text-risk-critical",
};

export function PredictForm({ tasks, thresholds }: { tasks: TaskName[]; thresholds: Partial<Record<TaskName, number | null>> }) {
  const [task, setTask] = useState<TaskName>(tasks[0] ?? "at_risk");
  const [form, setForm] = useState<Form>(DEFAULTS);
  const [explain, setExplain] = useState(true);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, start] = useTransition();

  const set = <K extends keyof Form>(k: K, v: Form[K]) => setForm((f) => ({ ...f, [k]: v }));

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    start(async () => {
      const body: Record<string, string | number> = {
        gender: form.gender,
        race_ethnicity: form.race_ethnicity,
        parental_level_of_education: form.parental_level_of_education,
        lunch: form.lunch,
        test_preparation_course: form.test_preparation_course,
      };
      for (const k of SCORES_FOR[task]) body[k] = form[k];
      try {
        const res = await fetch(`/api/predict/${TASK_SLUG[task]}?explain=${explain}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
        setResult({ task, data } as Result);
      } catch (err) {
        setResult(null);
        setError(err instanceof Error ? err.message : "Request failed");
      }
    });
  };

  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-5">
      <Card className="glass xl:col-span-2">
        <CardHeader>
          <CardTitle>Student</CardTitle>
          <CardDescription>Background attributes as recorded at enrolment. Scores only where the task needs them.</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={submit} className="flex flex-col gap-5">
            <Tabs value={task} onValueChange={(v) => { setTask(v as TaskName); setResult(null); }}>
              <TabsList aria-label="Prediction task" className="w-full">
                {tasks.map((t) => (
                  <TabsTrigger key={t} value={t} className="flex-1">
                    {TASK_LABEL[t].split(" ")[0]}
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>

            <SelectField id="gender" label="Gender" value={form.gender} options={GENDERS} onChange={(v) => set("gender", v)} />
            <SelectField id="race_ethnicity" label="Race / ethnicity" value={form.race_ethnicity} options={ETHNICITIES} onChange={(v) => set("race_ethnicity", v)} />
            <SelectField id="parental_level_of_education" label="Parental level of education" value={form.parental_level_of_education} options={PARENTAL_EDUCATION} onChange={(v) => set("parental_level_of_education", v)} />
            <SelectField id="lunch" label="Lunch" value={form.lunch} options={LUNCH} onChange={(v) => set("lunch", v)} hint="Free/reduced lunch is a proxy for household income." />
            <SelectField id="test_preparation_course" label="Test preparation course" value={form.test_preparation_course} options={TEST_PREP} onChange={(v) => set("test_preparation_course", v)} />

            {SCORES_FOR[task].map((k) => (
              <div key={k} className="flex flex-col gap-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor={k}>{k.replace("_", " ")}</Label>
                  <span className="font-mono text-sm tabular-nums">{form[k]}</span>
                </div>
                <Slider
                  id={k}
                  aria-label={k.replace("_", " ")}
                  min={0}
                  max={100}
                  step={1}
                  value={[form[k] as number]}
                  onValueChange={(v) => set(k, (Array.isArray(v) ? v[0] : v) as never)}
                />
              </div>
            ))}

            <label className="flex min-h-11 items-center gap-2 text-sm">
              <Checkbox checked={explain} onCheckedChange={(c) => setExplain(Boolean(c))} />
              Explain with SHAP
            </label>

            <Button type="submit" disabled={pending} className="min-h-11">
              {pending && <Loader2 className="size-4 animate-spin" aria-hidden="true" />}
              {pending ? "Scoring" : "Predict"}
            </Button>

            {error && (
              <p role="alert" className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm">
                <AlertCircle className="mt-0.5 size-4 shrink-0 text-destructive" aria-hidden="true" />
                {error}
              </p>
            )}
          </form>
        </CardContent>
      </Card>

      <div className="flex flex-col gap-4 xl:col-span-3" aria-live="polite">
        {!result && !pending && (
          <Card className="glass flex min-h-64 items-center justify-center">
            <CardContent className="text-center text-sm text-muted-foreground">
              Fill in the form and press Predict. The result and its explanation appear here.
            </CardContent>
          </Card>
        )}
        {result?.task === "at_risk" && <AtRiskResult data={result.data} threshold={thresholds.at_risk ?? result.data.threshold} />}
        {result?.task === "math_score" && <ScoreResult data={result.data} />}
        {result?.task === "performance_level" && <LevelResult data={result.data} />}
        {result?.data.explanation && <Explanation rows={result.data.explanation} kind={result.task} />}
      </div>
    </div>
  );
}

function SelectField({
  id,
  label,
  value,
  options,
  onChange,
  hint,
}: {
  id: string;
  label: string;
  value: string;
  options: readonly string[];
  onChange: (v: string) => void;
  hint?: string;
}) {
  return (
    <div className="flex flex-col gap-2">
      <Label htmlFor={id}>{label}</Label>
      <Select value={value} onValueChange={(v) => onChange(String(v))}>
        <SelectTrigger id={id} className="min-h-11 w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((o) => (
            <SelectItem key={o} value={o}>
              {o}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

function AtRiskResult({ data, threshold }: { data: AtRiskPrediction; threshold: number | null }) {
  const p = data.probability * 100;
  return (
    <Card className="glass">
      <CardHeader>
        <CardTitle>Probability of averaging below 60</CardTitle>
        <CardDescription>
          Decision threshold {fmt(threshold ?? data.threshold, 2)}, chosen so that at least 80% of at-risk students are caught on out-of-fold data.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col items-center gap-6 sm:flex-row sm:items-start">
        <ProgressRadial
          value={p}
          threshold={(threshold ?? data.threshold) * 100}
          size={200}
          label="Probability of averaging below 60"
          indicatorClassName={BAND_CLASS[data.risk_band]}
        >
          <div className="flex flex-col items-center">
            <span className="font-mono text-4xl font-semibold tabular-nums">{p.toFixed(1)}%</span>
            <span className={cn("text-xs font-medium uppercase", BAND_CLASS[data.risk_band])}>{data.risk_band}</span>
          </div>
        </ProgressRadial>
        <dl className="grid flex-1 grid-cols-2 gap-x-6 gap-y-3 text-sm">
          <dt className="text-muted-foreground">Flag for intervention</dt>
          <dd>
            <Badge variant={data.at_risk ? "destructive" : "secondary"}>{data.at_risk ? "yes" : "no"}</Badge>
          </dd>
          <dt className="text-muted-foreground">Risk band</dt>
          <dd className={cn("font-medium capitalize", BAND_CLASS[data.risk_band])}>{data.risk_band}</dd>
          {data.mitigated && (
            <>
              <dt className="text-muted-foreground">
                Flag with {data.mitigated.attribute}-equalised threshold ({fmt(data.mitigated.threshold, 2)})
              </dt>
              <dd>
                <Badge variant={data.mitigated.at_risk ? "destructive" : "secondary"}>{data.mitigated.at_risk ? "yes" : "no"}</Badge>
              </dd>
            </>
          )}
          <dt className="text-muted-foreground">Model version</dt>
          <dd className="font-mono text-xs">{data.model_version}</dd>
          <dd className="col-span-2 mt-2 rounded-md border border-border bg-muted/40 p-3 text-xs text-muted-foreground">
            This is a triage signal for prioritising support. It is not a judgement about the student.
          </dd>
        </dl>
      </CardContent>
    </Card>
  );
}

function ScoreResult({ data }: { data: ScorePrediction }) {
  return (
    <Card className="glass">
      <CardHeader>
        <CardTitle>Expected math score</CardTitle>
        <CardDescription>Ridge regression on background plus reading and writing. Hold-out RMSE is about 5.4 points.</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex items-baseline gap-3">
          <span className="font-mono text-5xl font-semibold tabular-nums">{data.prediction.toFixed(1)}</span>
          <span className="text-muted-foreground">/ 100</span>
          <span className="ml-auto font-mono text-xs text-muted-foreground">v{data.model_version}</span>
        </div>
        {data.lower != null && data.upper != null && data.confidence != null && (
          <IntervalBar lower={data.lower} upper={data.upper} point={data.prediction} confidence={data.confidence} />
        )}
      </CardContent>
    </Card>
  );
}

function IntervalBar({ lower, upper, point, confidence }: { lower: number; upper: number; point: number; confidence: number }) {
  const pct = (v: number) => `${Math.min(100, Math.max(0, v))}%`;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="relative h-2 w-full rounded-full bg-muted">
        <div className="absolute h-2 rounded-full bg-primary/30" style={{ left: pct(lower), width: pct(upper - lower) }} />
        <div className="absolute top-1/2 h-3.5 w-0.5 -translate-y-1/2 bg-primary" style={{ left: pct(point) }} />
      </div>
      <div className="flex justify-between font-mono text-xs text-muted-foreground">
        <span>{lower.toFixed(1)}</span>
        <span>{Math.round(confidence * 100)}% conformal interval</span>
        <span>{upper.toFixed(1)}</span>
      </div>
    </div>
  );
}

function LevelResult({ data }: { data: LevelPrediction }) {
  const order = ["low", "medium", "high"] as const;
  const colour = { low: "bg-risk-critical", medium: "bg-risk-moderate", high: "bg-risk-low" };
  return (
    <Card className="glass">
      <CardHeader>
        <CardTitle>Performance level</CardTitle>
        <CardDescription>Predicted tier from all three scores. This task is a leakage case study; see the leaderboard note.</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <span className="font-mono text-4xl font-semibold uppercase">{data.label}</span>
        <ul className="flex flex-col gap-2">
          {order.map((k) => (
            <li key={k} className="flex items-center gap-3 text-sm">
              <span className="w-16 capitalize">{k}</span>
              <span className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
                <span className={cn("block h-full rounded-full", colour[k])} style={{ width: pct(data.probabilities[k], 1) }} />
              </span>
              <span className="w-14 text-right font-mono text-xs">{pct(data.probabilities[k])}</span>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

function Explanation({ rows, kind }: { rows: Contribution[]; kind: TaskName }) {
  const unit = kind === "math_score" ? "score points" : kind === "at_risk" ? "log-odds of being at risk" : "log-odds of the predicted tier";
  return (
    <Card className="glass">
      <CardHeader>
        <CardTitle>Why</CardTitle>
        <CardDescription>
          SHAP contributions summed to the original features, in {unit}. Red pushes the prediction up, violet pushes it down.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ContributionBars data={[...rows].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))} />
      </CardContent>
    </Card>
  );
}
