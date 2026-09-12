"use client";

import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { cn } from "cn";
import { LUNCH, PARENTAL_EDUCATION, TEST_PREP, GENDERS, type AtRiskPrediction } from "@/lib/api";

type Form = {
  gender: string;
  race_ethnicity: string;
  parental_level_of_education: string;
  lunch: string;
  test_preparation_course: string;
};

const START: Form = {
  gender: "female",
  race_ethnicity: "group C",
  parental_level_of_education: "high school",
  lunch: "free/reduced",
  test_preparation_course: "none",
};

const BAND: Record<AtRiskPrediction["risk_band"], string> = {
  low: "bg-risk-low",
  moderate: "bg-risk-moderate",
  high: "bg-risk-high",
  critical: "bg-risk-critical",
};

function Chips({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: readonly string[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="eyebrow text-muted-foreground">{label}</legend>
      <div className="flex flex-wrap gap-2" role="radiogroup" aria-label={label}>
        {options.map((o) => {
          const active = o === value;
          return (
            <button
              key={o}
              type="button"
              role="radio"
              aria-checked={active}
              onClick={() => onChange(o)}
              className={cn(
                "min-h-10 rounded-full border px-4 text-sm transition-colors duration-150",
                "focus-visible:outline-2 focus-visible:outline-ring focus-visible:outline-offset-2",
                active
                  ? "border-primary bg-primary text-primary-foreground"
                  : "border-border bg-transparent text-foreground hover:border-foreground/40",
              )}
            >
              {o}
            </button>
          );
        })}
      </div>
    </fieldset>
  );
}

/** Scores one student as the chips change. Calls the same proxy route as the Predict page. */
export function LiveDemo() {
  const [form, setForm] = useState<Form>(START);
  const [result, setResult] = useState<AtRiskPrediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const ctrl = new AbortController();
    const t = setTimeout(async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch("/api/predict/at-risk", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(form),
          signal: ctrl.signal,
        });
        if (!res.ok) throw new Error(`The API answered ${res.status}`);
        setResult((await res.json()) as AtRiskPrediction);
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          setError("The API is waking up or offline. Try again in a moment.");
        }
      } finally {
        setLoading(false);
      }
    }, 150);
    return () => {
      clearTimeout(t);
      ctrl.abort();
    };
  }, [form]);

  const set = (k: keyof Form) => (v: string) => setForm((f) => ({ ...f, [k]: v }));
  const p = result ? result.probability * 100 : 0;

  return (
    <div className="grid grid-cols-1 gap-8 lg:grid-cols-5">
      <div className="flex flex-col gap-6 lg:col-span-3">
        <Chips label="Lunch programme" options={LUNCH} value={form.lunch} onChange={set("lunch")} />
        <Chips label="Test preparation course" options={TEST_PREP} value={form.test_preparation_course} onChange={set("test_preparation_course")} />
        <Chips label="Parental education" options={PARENTAL_EDUCATION} value={form.parental_level_of_education} onChange={set("parental_level_of_education")} />
        <Chips label="Gender" options={GENDERS} value={form.gender} onChange={set("gender")} />
      </div>

      <div className="flex flex-col justify-between gap-6 rounded-2xl border border-border bg-card p-6 lg:col-span-2" aria-live="polite">
        <div>
          <p className="eyebrow text-muted-foreground">Probability of averaging below 60</p>
          <div className="mt-3 flex items-baseline gap-3">
            <span className="font-display text-6xl font-medium tabular-nums">
              {result ? p.toFixed(0) : loading ? "…" : "–"}
              <span className="text-3xl text-muted-foreground">%</span>
            </span>
            {loading && <Loader2 className="size-4 animate-spin text-muted-foreground" aria-label="Scoring" />}
          </div>
          <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-muted">
            <div
              className={cn("h-2 rounded-full transition-[width] duration-500 ease-out", result ? BAND[result.risk_band] : "bg-muted")}
              style={{ width: `${Math.max(2, p)}%` }}
            />
          </div>
          {result && (
            <p className="mt-2 text-xs text-muted-foreground">
              <span className="capitalize">{result.risk_band}</span> risk band, decision threshold {result.threshold.toFixed(2)}
            </p>
          )}
        </div>

        {result && (
          <dl className="grid grid-cols-2 gap-x-4 gap-y-3 border-t border-border pt-5 text-sm">
            <dt className="text-muted-foreground">Global threshold</dt>
            <dd className="font-medium">{result.at_risk ? "flag for support" : "no flag"}</dd>
            {result.mitigated && (
              <>
                <dt className="text-muted-foreground">Recall-equalised threshold ({result.mitigated.threshold.toFixed(2)})</dt>
                <dd className="font-medium">{result.mitigated.at_risk ? "flag for support" : "no flag"}</dd>
              </>
            )}
          </dl>
        )}
        {error && <p className="text-sm text-destructive">{error}</p>}
        <p className="text-xs text-muted-foreground">
          A triage signal for prioritising support, never a judgement about a student. Full form with SHAP explanations in the app.
        </p>
      </div>
    </div>
  );
}
