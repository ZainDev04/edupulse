import Image from "next/image";
import Link from "next/link";
import {
  ArrowRight,
  ArrowUpRight,
  Activity,
  BarChart3,
  GitBranch,
  Scale,
  ShieldCheck,
  Sparkles,
  Target,
} from "lucide-react";
import { api, fmt } from "@/lib/api";
import { LiveDemo } from "@/components/marketing/live-demo";
import { Reveal } from "@/components/marketing/reveal";
import { Button } from "@/components/ui/button";

// Numbers from the committed model cards; replaced by live API values when the API is up.
const FALLBACK = {
  auc: 0.699,
  recall: 0.737,
  r2: 0.882,
  coverage: 0.885,
  gapBefore: 0.447,
  gapAfter: 0.031,
  recallAfter: 0.877,
  version: "1.4.0",
};

export default async function LandingPage() {
  const [models, fair, health] = await Promise.all([api.models(), api.fairness("at_risk"), api.health()]);
  const atRisk = models?.at_risk;
  const math = models?.math_score;
  const mit = fair?.mitigated;
  const gapBefore = fair?.summary?.lunch?.tpr_gap ?? FALLBACK.gapBefore;
  const gapAfter = mit?.summary?.lunch?.tpr_gap ?? FALLBACK.gapAfter;
  const n = {
    auc: atRisk?.test_metrics.roc_auc ?? FALLBACK.auc,
    recall: atRisk?.test_metrics.recall ?? FALLBACK.recall,
    recallAfter: mit?.overall_after.recall ?? FALLBACK.recallAfter,
    r2: math?.test_metrics.r2 ?? FALLBACK.r2,
    coverage: math?.test_metrics.interval_coverage ?? FALLBACK.coverage,
    version: health?.version ?? FALLBACK.version,
  };

  return (
    <>
      {/* Hero */}
      <section className="relative overflow-hidden px-5 pb-24 pt-36 sm:px-8 sm:pt-44">
        <div
          aria-hidden="true"
          className="lp-glow pointer-events-none absolute inset-0"
          style={{
            background:
              "radial-gradient(60% 50% at 70% 10%, rgba(139,92,246,0.28) 0%, rgba(139,92,246,0) 60%), radial-gradient(40% 40% at 15% 80%, rgba(56,189,248,0.12) 0%, rgba(56,189,248,0) 60%)",
          }}
        />
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 opacity-[0.06]"
          style={{
            backgroundImage: "linear-gradient(rgba(255,255,255,.5) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.5) 1px, transparent 1px)",
            backgroundSize: "72px 72px",
            maskImage: "radial-gradient(70% 60% at 50% 30%, black, transparent)",
          }}
        />
        <div className="relative mx-auto flex max-w-4xl flex-col items-center text-center">
          <p className="lp-in lp-d0 eyebrow text-primary">Student performance intelligence</p>
          <h1 className="lp-in lp-d1 mt-6 max-w-4xl text-balance font-display text-5xl font-medium leading-[1.05] tracking-tight sm:text-6xl md:text-7xl">
            <em className="italic">Catch the student</em> who will struggle, before the <em className="italic">first exam</em>
            <span className="text-primary">.</span>
          </h1>
          <p className="lp-in lp-d2 mt-7 max-w-2xl text-lg leading-relaxed text-white/70">
            EduPulse turns a public exam dataset into a complete early-warning system: ten model families benchmarked,
            a threshold set from a recall target, every prediction explained, every group audited, and the whole
            thing served from one API.
          </p>
          <div className="lp-in lp-d3 mt-9 flex flex-wrap items-center justify-center gap-3">
            <Button render={<a href="#demo" />} nativeButton={false} size="lg">
              Try the live demo <ArrowRight className="size-4" aria-hidden="true" />
            </Button>
            <Button render={<Link href="/overview" />} nativeButton={false} size="lg" variant="outline" className="border-white/20 bg-transparent text-white hover:bg-white/10 hover:text-white">
              Open the app
            </Button>
          </div>
          <p className="lp-in lp-d4 mt-5 text-xs text-white/40">
            v{n.version}, 63 tests, MIT licence. Free-tier hosting: the first request after idle can take a moment.
          </p>
        </div>

        <dl className="lp-in lp-d5 relative mx-auto mt-20 grid max-w-5xl grid-cols-2 gap-px overflow-hidden rounded-2xl border border-white/10 bg-white/10 md:grid-cols-4">
          {[
            { k: "Recall gap across lunch groups", v: `${fmt(gapBefore, 3)} to ${fmt(gapAfter, 3)}`, s: "after per-group thresholds" },
            { k: "At-risk recall, hold-out", v: `${Math.round(n.recall * 100)}% to ${Math.round(n.recallAfter * 100)}%`, s: "same model, fairer cut-off" },
            { k: "Math-score interval coverage", v: `${(n.coverage * 100).toFixed(1)}%`, s: "90% conformal target" },
            { k: "Model families benchmarked", v: "10", s: "repeated 5x2 cross-validation" },
          ].map((s) => (
            <div key={s.k} className="flex flex-col gap-1 bg-[#0b0e1a] p-6">
              <dt className="text-xs text-white/50">{s.k}</dt>
              <dd className="font-display text-3xl tabular-nums">{s.v}</dd>
              <dd className="text-xs text-white/40">{s.s}</dd>
            </div>
          ))}
        </dl>
      </section>

      {/* 01 The problem */}
      <section id="problem" className="light bg-background px-5 py-24 text-foreground sm:px-8 sm:py-32">
        <div className="mx-auto grid max-w-7xl gap-12 lg:grid-cols-12">
          <Reveal className="lg:col-span-5">
            <p className="eyebrow text-primary">01. The problem</p>
            <h2 className="mt-4 font-display text-4xl font-medium leading-tight sm:text-5xl">
              A 96% accurate model that <em className="italic">answered the wrong question</em>.
            </h2>
          </Reveal>
          <Reveal className="flex flex-col gap-5 text-base leading-relaxed text-muted-foreground lg:col-span-7 lg:pt-12" delay={120}>
            <p>
              The original coursework predicted a student&rsquo;s performance tier from their three exam scores. It scored
              96%, and it meant nothing: the tier is defined as the average of those scores, cut at 60 and 80. Any model
              just re-learns two cut-offs.
            </p>
            <p>
              EduPulse keeps that task as a documented leakage case study and asks the question a school can act on:
              which students are likely to average below 60, judged from background alone, before any exam is taken? The
              number is lower, and the number means something.
            </p>
            <div className="mt-2 grid grid-cols-2 gap-4">
              <div className="rounded-xl border border-border bg-card p-5">
                <p className="eyebrow text-muted-foreground">Coursework task</p>
                <p className="mt-2 font-display text-3xl">98.5%</p>
                <p className="text-xs text-muted-foreground">accuracy, target leakage</p>
              </div>
              <div className="rounded-xl border border-primary/40 bg-card p-5">
                <p className="eyebrow text-primary">Early-warning task</p>
                <p className="mt-2 font-display text-3xl">{fmt(n.auc, 3)}</p>
                <p className="text-xs text-muted-foreground">ROC-AUC from five background fields</p>
              </div>
            </div>
          </Reveal>
        </div>
      </section>

      {/* 02 What it does */}
      <section className="light border-t border-border bg-card px-5 py-24 text-foreground sm:px-8 sm:py-32">
        <div className="mx-auto max-w-7xl">
          <Reveal className="max-w-2xl">
            <p className="eyebrow text-primary">02. What it does</p>
            <h2 className="mt-4 font-display text-4xl font-medium leading-tight sm:text-5xl">
              Three questions, <em className="italic">one pipeline</em>.
            </h2>
          </Reveal>
          <ul className="mt-14 grid gap-8 md:grid-cols-3">
            {[
              {
                icon: Target,
                title: "Early warning",
                body: "Probability that a student averages below 60, from lunch programme, test preparation, parental education, ethnicity and gender. Threshold chosen so at least 80% of at-risk students are caught.",
              },
              {
                icon: BarChart3,
                title: "Score prediction with a guarantee",
                body: "Expected math score from reading, writing and background, with a conformal interval calibrated on out-of-fold residuals. Nominal 90%, measured 88.5% on held-out students.",
              },
              {
                icon: Sparkles,
                title: "Explain, audit, watch",
                body: "SHAP contributions on every prediction, a subgroup fairness audit with a before-and-after mitigation table, and drift monitoring on the live prediction stream.",
              },
            ].map((f, i) => (
              <Reveal as="li" key={f.title} delay={i * 100} className="flex flex-col gap-4 rounded-2xl border border-border bg-background p-7">
                <span className="flex size-12 items-center justify-center rounded-full bg-primary/10 text-primary">
                  <f.icon className="size-5" aria-hidden="true" />
                </span>
                <h3 className="font-display text-2xl">{f.title}</h3>
                <span className="h-0.5 w-10 bg-primary" aria-hidden="true" />
                <p className="text-sm leading-relaxed text-muted-foreground">{f.body}</p>
              </Reveal>
            ))}
          </ul>
        </div>
      </section>

      {/* 03 Results */}
      <section id="results" className="light bg-background px-5 py-24 text-foreground sm:px-8 sm:py-32">
        <div className="mx-auto max-w-7xl">
          <Reveal className="max-w-2xl">
            <p className="eyebrow text-primary">03. Results</p>
            <h2 className="mt-4 font-display text-4xl font-medium leading-tight sm:text-5xl">
              Honest numbers, <em className="italic">on held-out students</em>.
            </h2>
            <p className="mt-5 text-base leading-relaxed text-muted-foreground">
              200 students are held out before anything is trained. Model selection runs on repeated cross-validation
              of the other 800. What you see below was never used to pick a model.
            </p>
          </Reveal>
          <div className="mt-14 grid gap-8 lg:grid-cols-2">
            <Reveal className="overflow-hidden rounded-2xl border border-border bg-card">
              <Image src="/figures/fairness_mitigation.png" alt="Recall and selection rate per lunch group under the global and per-group thresholds" width={1860} height={780} className="w-full" />
              <div className="p-6">
                <h3 className="font-display text-2xl">Recall equalised across lunch groups</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                  One global threshold flagged 95% of free/reduced-lunch students and caught only 52% of at-risk
                  students on standard lunch. Per-group thresholds bring both groups to about 0.87 recall and cut the
                  gap from {fmt(gapBefore, 3)} to {fmt(gapAfter, 3)}. The model is unchanged; only the cut-off moves.
                </p>
              </div>
            </Reveal>
            <Reveal delay={120} className="overflow-hidden rounded-2xl border border-border bg-card">
              <Image src="/figures/prediction_intervals.png" alt="Hold-out students sorted by predicted math score with the 90% conformal band" width={1860} height={830} className="w-full" />
              <div className="p-6">
                <h3 className="font-display text-2xl">Intervals that say how sure the model is</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                  R² {fmt(n.r2, 3)} for the math score, and every prediction carries a symmetric band of about 9 points.
                  The band contains the true score {(n.coverage * 100).toFixed(1)}% of the time on held-out students,
                  against a 90% target.
                </p>
              </div>
            </Reveal>
          </div>
          <Reveal className="mt-10 overflow-x-auto rounded-2xl border border-border bg-card">
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-muted-foreground">
                <tr className="border-b border-border">
                  <th className="px-6 py-4 font-medium">Task</th>
                  <th className="px-6 py-4 font-medium">Best of 10 families</th>
                  <th className="px-6 py-4 font-medium">Cross-validation</th>
                  <th className="px-6 py-4 font-medium">Hold-out</th>
                </tr>
              </thead>
              <tbody className="font-mono text-xs">
                <tr className="border-b border-border">
                  <td className="px-6 py-4 font-sans text-sm">At-risk early warning</td>
                  <td className="px-6 py-4">logistic regression, tuned C</td>
                  <td className="px-6 py-4">ROC-AUC 0.742 ± 0.034</td>
                  <td className="px-6 py-4">ROC-AUC {fmt(n.auc, 3)}, recall {Math.round(n.recall * 100)}% at the global threshold, {Math.round(n.recallAfter * 100)}% equalised</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="px-6 py-4 font-sans text-sm">Math-score prediction</td>
                  <td className="px-6 py-4">ridge, tuned alpha</td>
                  <td className="px-6 py-4">R² 0.867 ± 0.019</td>
                  <td className="px-6 py-4">R² {fmt(n.r2, 3)}, RMSE 5.36, 90% interval covers {(n.coverage * 100).toFixed(1)}%</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 font-sans text-sm">Performance level (leakage study)</td>
                  <td className="px-6 py-4">hist gradient boosting</td>
                  <td className="px-6 py-4">macro-F1 0.971</td>
                  <td className="px-6 py-4">accuracy 0.985, by construction</td>
                </tr>
              </tbody>
            </table>
          </Reveal>
        </div>
      </section>

      {/* 04 Live demo */}
      <section id="demo" className="border-t border-white/10 bg-[#0b0e1a] px-5 py-24 text-white sm:px-8 sm:py-32">
        <div className="mx-auto max-w-7xl">
          <Reveal className="max-w-2xl">
            <p className="eyebrow text-primary">04. Live demo</p>
            <h2 className="mt-4 font-display text-4xl font-medium leading-tight sm:text-5xl">
              Score a student <em className="italic">as you change the facts</em>.
            </h2>
            <p className="mt-5 text-base leading-relaxed text-white/70">
              Five background fields go to the registered pipeline through the REST API. Both flags come back: the
              global threshold and the lunch-equalised one.
            </p>
          </Reveal>
          <Reveal className="mt-12" delay={100}>
            <LiveDemo />
          </Reveal>
        </div>
      </section>

      {/* 05 How it works */}
      <section id="how" className="light bg-background px-5 py-24 text-foreground sm:px-8 sm:py-32">
        <div className="mx-auto grid max-w-7xl gap-12 lg:grid-cols-12">
          <Reveal className="lg:col-span-5">
            <p className="eyebrow text-primary">05. How it works</p>
            <h2 className="mt-4 font-display text-4xl font-medium leading-tight sm:text-5xl">
              One command, <em className="italic">eight steps</em>, one pipeline object.
            </h2>
            <p className="mt-5 text-base leading-relaxed text-muted-foreground">
              Every persisted model is a single scikit-learn Pipeline: feature engineering, encoding and the estimator
              together, so the API accepts raw JSON and there is no train/serve skew.
            </p>
            <div className="mt-8 flex flex-wrap gap-2 text-xs text-muted-foreground">
              {["Python", "scikit-learn", "Optuna", "SHAP", "MLflow", "FastAPI", "Next.js", "Streamlit", "Docker", "GitHub Actions"].map((t) => (
                <span key={t} className="rounded-full border border-border px-3 py-1">{t}</span>
              ))}
            </div>
          </Reveal>
          <ol className="flex flex-col lg:col-span-7">
            {[
              ["Validate", "The CSV is checked against a declared schema before anything else runs."],
              ["Benchmark", "Ten model families scored with repeated stratified cross-validation on the training split."],
              ["Tune", "The best family is tuned with Optuna. If tuning does not beat the defaults, the defaults stay."],
              ["Set the cut-off", "The highest threshold that still reaches the target recall on out-of-fold probabilities, then the same rule per lunch group."],
              ["Calibrate", "For regression, out-of-fold residuals give a finite-sample-corrected conformal half-width."],
              ["Evaluate and explain", "Hold-out metrics, SHAP and permutation importance, subgroup audit before and after mitigation."],
              ["Register", "Pipeline, metadata and a generated model card, versioned on disk and mirrored in MLflow."],
              ["Serve and watch", "FastAPI, this web app and a Streamlit dashboard share one service; the API monitors its own prediction stream for drift."],
            ].map(([title, body], i) => (
              <Reveal as="li" key={title} delay={i * 60} className="grid grid-cols-[3rem_1fr] gap-4 border-b border-border py-5 last:border-b-0">
                <span className="font-mono text-sm text-primary">{String(i + 1).padStart(2, "0")}</span>
                <div>
                  <h3 className="font-display text-xl">{title}</h3>
                  <p className="mt-1 text-sm leading-relaxed text-muted-foreground">{body}</p>
                </div>
              </Reveal>
            ))}
          </ol>
        </div>
      </section>

      {/* 06 Responsible use */}
      <section id="fairness" className="light border-t border-border bg-card px-5 py-24 text-foreground sm:px-8 sm:py-32">
        <div className="mx-auto max-w-7xl">
          <Reveal className="max-w-2xl">
            <p className="eyebrow text-primary">06. Responsible use</p>
            <h2 className="mt-4 font-display text-4xl font-medium leading-tight sm:text-5xl">
              A triage signal, <em className="italic">never a verdict</em>.
            </h2>
          </Reveal>
          <div className="mt-14 grid gap-8 md:grid-cols-3">
            {[
              { icon: Scale, title: "Audited, not hidden", body: "Sensitive attributes are inputs, and their effect is measured and published in every model card: selection rate, recall, false-positive rate and precision per group." },
              { icon: ShieldCheck, title: "Mitigated, with the cost shown", body: "Per-group thresholds equalise recall across lunch groups. The before-and-after table, including overall precision and the flagged share, is part of the audit." },
              { icon: Activity, title: "Watched in production", body: "Population stability per input and a KS test on the output, against the training population, so a change in who is being scored is visible before it matters." },
            ].map((f, i) => (
              <Reveal key={f.title} delay={i * 100} className="flex flex-col gap-4">
                <span className="flex size-12 items-center justify-center rounded-full bg-primary/10 text-primary">
                  <f.icon className="size-5" aria-hidden="true" />
                </span>
                <h3 className="font-display text-2xl">{f.title}</h3>
                <p className="text-sm leading-relaxed text-muted-foreground">{f.body}</p>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* Closing */}
      <section className="relative overflow-hidden px-5 py-28 text-center sm:px-8 sm:py-36">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0"
          style={{ background: "radial-gradient(50% 60% at 50% 100%, rgba(139,92,246,0.25) 0%, rgba(139,92,246,0) 70%)" }}
        />
        <Reveal className="relative mx-auto flex max-w-3xl flex-col items-center">
          <h2 className="font-display text-4xl font-medium leading-tight sm:text-5xl">
            Read the code, <em className="italic">then run it</em>.
          </h2>
          <p className="mt-5 max-w-xl text-base leading-relaxed text-white/70">
            One repository holds the library, the API, both front ends, the notebooks, the model cards and the CI that
            trains a model on every push.
          </p>
          <div className="mt-9 flex flex-wrap justify-center gap-3">
            <Button render={<a href="https://github.com/ZainDev04/edupulse" target="_blank" rel="noreferrer" />} nativeButton={false} size="lg">
              <GitBranch className="size-4" aria-hidden="true" /> View on GitHub <ArrowUpRight className="size-4" aria-hidden="true" />
            </Button>
            <Button render={<Link href="/predict" />} nativeButton={false} size="lg" variant="outline" className="border-white/20 bg-transparent text-white hover:bg-white/10 hover:text-white">
              Full prediction form
            </Button>
          </div>
        </Reveal>
      </section>
    </>
  );
}
