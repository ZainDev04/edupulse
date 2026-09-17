import Link from "next/link";
import type { LucideIcon } from "lucide-react";
import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  CheckCircle2,
  Gauge,
  GraduationCap,
  Scale,
  ShieldAlert,
  Sparkles,
  Target,
  Users,
} from "lucide-react";
import { api, fmt, pct, titleCase, type ModelInfo, type TaskName, TASK_LABEL } from "@/lib/api";
import { RankedBars } from "@/components/charts/bar-charts";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { OfflineNotice } from "@/components/offline-notice";
import { OVERVIEW_THEME_KEY, OverviewThemeToggle } from "@/components/overview-theme-toggle";

const PILL = "rounded-full px-5 text-sm font-medium focus-visible:ring-4 focus-visible:ring-ring/60";

const CAPABILITIES: { href: string; title: string; sub: string; icon: LucideIcon }[] = [
  { href: "/predict", title: "Score a student", sub: "risk band plus interval", icon: Target },
  { href: "/explain", title: "Explain a prediction", sub: "SHAP per feature", icon: Sparkles },
  { href: "/fairness", title: "Audit fairness", sub: "recall gap 0.45 to 0.03", icon: Scale },
  { href: "/monitoring", title: "Watch for drift", sub: "PSI and KS on live traffic", icon: Activity },
];

function headline(kind: string, m: Record<string, number>): { label: string; value: string; secondary: string } {
  if (kind === "binary") return { label: "ROC-AUC", value: fmt(m.roc_auc), secondary: `recall ${pct(m.recall, 0)}` };
  if (kind === "regression") return { label: "R²", value: fmt(m.r2), secondary: `RMSE ${fmt(m.rmse, 2)}` };
  return { label: "Accuracy", value: pct(m.accuracy), secondary: `macro-F1 ${fmt(m.f1_macro)}` };
}

export default async function OverviewPage() {
  const [stats, models, health] = await Promise.all([api.stats(), api.models(), api.health()]);
  if (!stats || !models) return <OfflineNotice />;

  const atRisk = models.at_risk;
  const lunch = stats.by_attribute.lunch ?? [];
  const prep = stats.by_attribute.test_preparation_course ?? [];
  const parental = [...(stats.by_attribute.parental_level_of_education ?? [])].sort(
    (a, b) => b.at_risk_rate - a.at_risk_rate,
  );
  const tasks = Object.keys(models) as TaskName[];

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-10">
      {/* Applies the stored overview theme before hydration so there is no flash */}
      <script
        dangerouslySetInnerHTML={{
          __html: `try{if(localStorage.getItem(${JSON.stringify(OVERVIEW_THEME_KEY)})==="light"){var r=document.documentElement;r.classList.remove("dark");r.classList.add("light")}}catch(e){}`,
        }}
      />
      {/* Hero */}
      <section className="ov-aurora relative overflow-hidden rounded-[28px] border border-white/10 px-6 py-14 text-center sm:px-10 sm:py-20">
        <div className="absolute top-4 right-4">
          <OverviewThemeToggle />
        </div>
        <div className="relative mx-auto flex max-w-3xl flex-col items-center gap-6">
          <div className="flex flex-wrap items-center justify-center gap-2">
            <span className="rounded-full bg-sky-500/90 px-3 py-1 text-xs font-semibold text-white">
              {health ? `Release ${health.version}` : "Release"}
            </span>
            <span className="rounded-full border border-white/20 bg-white/10 px-3 py-1 text-xs font-medium text-white/90 backdrop-blur">
              {tasks.length} models registered, {stats.n_students.toLocaleString()} students
            </span>
          </div>
          <h1 className="ov-headline text-4xl text-white sm:text-6xl">
            Student performance
            <br />
            intelligence
          </h1>
          <p className="max-w-2xl text-base text-white/80 sm:text-lg">
            The at-risk model flags students likely to average below 60 from background information alone,
            before any exam is taken. Every prediction comes with a SHAP explanation and every model ships
            with a fairness audit.
          </p>
          <div className="flex flex-wrap justify-center gap-3">
            <Button
              render={<Link href="/predict" />}
              nativeButton={false}
              size="lg"
              className={`${PILL} h-11 bg-violet-500 text-white hover:bg-violet-400`}
            >
              Score a student <ArrowRight className="size-4" aria-hidden="true" />
            </Button>
            <Button
              render={<Link href="/leaderboard" />}
              nativeButton={false}
              size="lg"
              className={`${PILL} h-11 bg-white text-slate-900 hover:bg-white/90`}
            >
              Model leaderboard
            </Button>
          </div>
        </div>
      </section>

      {/* Key figures */}
      <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4" aria-label="Key figures">
        <Tile tone="violet" icon={Users} value={stats.n_students.toLocaleString()} label="Students" hint="rows after cleaning" />
        <Tile tone="amber" icon={ShieldAlert} value={pct(stats.at_risk_rate)} label="At-risk rate" hint="average score below 60" />
        <Tile tone="blue" icon={GraduationCap} value={stats.average_score.toFixed(1)} label="Average score" hint="mean of math, reading, writing" />
        <Tile
          tone="green"
          icon={Gauge}
          value={atRisk ? fmt(atRisk.test_metrics.roc_auc) : "n/a"}
          label="At-risk ROC-AUC"
          hint={atRisk ? `${titleCase(atRisk.model)}, hold-out` : "not trained"}
        />
      </section>

      {/* Registered models */}
      <section className="flex flex-col gap-6" aria-labelledby="models-heading">
        <div className="flex flex-col items-center gap-3 text-center">
          <Eyebrow>Registered models</Eyebrow>
          <h2 id="models-heading" className="ov-headline text-3xl sm:text-4xl">
            Three tasks, <span className="ov-gradient-text">one pipeline</span>
          </h2>
          <p className="max-w-xl text-sm text-muted-foreground">
            Latest version per task. Hold-out metrics are computed on 200 students the models never saw.
          </p>
        </div>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          {tasks.map((t) => (
            <ModelCard key={t} task={t} info={models[t]} />
          ))}
        </div>
      </section>

      {/* What predicts risk */}
      <section className="grid grid-cols-1 gap-4 lg:grid-cols-5" aria-labelledby="risk-heading">
        <Card className="ov-tile ov-tile-violet rounded-3xl lg:col-span-2">
          <CardHeader className="relative">
            <CardTitle id="risk-heading" className="ov-headline text-xl">
              What predicts risk
            </CardTitle>
            <CardDescription>At-risk rate by group in the training data.</CardDescription>
          </CardHeader>
          <CardContent className="relative flex flex-col gap-5">
            <ChipGroup title="Lunch" rows={lunch} />
            <ChipGroup title="Test preparation" rows={prep} />
            <p className="text-xs text-muted-foreground">
              Free or reduced lunch and no preparation course are the two strongest background signals the
              at-risk model picks up.
            </p>
          </CardContent>
        </Card>

        <Card className="ov-tile ov-tile-blue rounded-3xl lg:col-span-3">
          <CardHeader className="relative">
            <CardTitle className="ov-headline text-xl">At-risk rate by parental education</CardTitle>
            <CardDescription>
              Dashed line is the overall rate ({pct(stats.at_risk_rate)}). Higher parental education tracks
              lower risk.
            </CardDescription>
          </CardHeader>
          <CardContent className="relative">
            <RankedBars
              data={parental.map((g) => ({ name: g.group, rate: g.at_risk_rate, n: g.n }))}
              valueKey="rate"
              format="pct"
              domain={[0, 0.6]}
              reference={{ value: stats.at_risk_rate, label: "overall" }}
            />
          </CardContent>
        </Card>
      </section>

      {/* Where to go next */}
      <section className="flex flex-col gap-6 rounded-[28px] border border-border bg-card/60 px-6 py-10" aria-labelledby="next-heading">
        <div className="flex flex-col items-center gap-3 text-center">
          <Eyebrow>Explore</Eyebrow>
          <h2 id="next-heading" className="ov-headline text-3xl sm:text-4xl">
            Built for <span className="ov-gradient-text">accountable</span> predictions
          </h2>
        </div>
        <ul className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {CAPABILITIES.map(({ href, title, sub, icon: Icon }) => (
            <li key={href}>
              <Link
                href={href}
                className="group flex h-full flex-col items-center gap-3 rounded-2xl border border-transparent px-4 py-6 text-center transition-colors hover:border-border hover:bg-muted/40 focus-visible:border-ring focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-ring/50"
              >
                <span className="flex size-12 items-center justify-center rounded-full bg-gradient-to-br from-violet-500 to-fuchsia-500 text-white shadow-lg shadow-violet-500/30">
                  <Icon className="size-5" aria-hidden="true" />
                </span>
                <span className="ov-headline text-base">{title}</span>
                <span className="text-sm text-violet-700 dark:text-violet-300">{sub}</span>
                <span className="mt-auto inline-flex items-center gap-1 text-xs text-muted-foreground group-hover:text-foreground">
                  Open <ArrowUpRight className="size-3.5" aria-hidden="true" />
                </span>
              </Link>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

function Eyebrow({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded-full border border-border bg-card px-3 py-1 text-[11px] font-medium tracking-[0.18em] text-muted-foreground uppercase">
      {children}
    </span>
  );
}

function Tile({
  tone,
  icon: Icon,
  value,
  label,
  hint,
}: {
  tone: "violet" | "blue" | "green" | "amber";
  icon: LucideIcon;
  value: string;
  label: string;
  hint: string;
}) {
  return (
    <div className={`ov-tile ov-tile-${tone} flex min-h-40 flex-col justify-between rounded-3xl p-6`}>
      <div className="relative flex items-center justify-between">
        <span className="text-sm font-medium text-foreground/80">{label}</span>
        <span className="flex size-9 items-center justify-center rounded-full bg-foreground/10">
          <Icon className="size-4 text-foreground" aria-hidden="true" />
        </span>
      </div>
      <div className="relative">
        <div className="ov-headline ov-gradient-text text-5xl tabular-nums">{value}</div>
        <div className="mt-1 text-xs text-muted-foreground">{hint}</div>
      </div>
    </div>
  );
}

function ModelCard({ task, info }: { task: TaskName; info: ModelInfo }) {
  const h = headline(info.kind, info.test_metrics);
  return (
    <Card className="ov-tile rounded-3xl">
      <CardHeader className="relative gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle className="ov-headline text-lg">{TASK_LABEL[task]}</CardTitle>
          {info.leakage_note && (
            <Badge variant="outline" className="text-[10px]">
              leakage case study
            </Badge>
          )}
        </div>
        <CardDescription>{titleCase(info.model)}</CardDescription>
      </CardHeader>
      <CardContent className="relative flex flex-col gap-4">
        <dl className="grid grid-cols-2 gap-3 text-sm">
          <div className="rounded-xl bg-foreground/5 p-3">
            <dt className="text-xs text-muted-foreground">Cross-validation {info.cv_metric}</dt>
            <dd className="ov-headline mt-1 text-2xl tabular-nums">{fmt(info.cv_score)}</dd>
          </div>
          <div className="rounded-xl bg-foreground/5 p-3">
            <dt className="text-xs text-muted-foreground">Hold-out {h.label}</dt>
            <dd className="ov-headline mt-1 text-2xl tabular-nums">{h.value}</dd>
            <dd className="font-mono text-xs text-muted-foreground">{h.secondary}</dd>
          </div>
        </dl>
        <div className="flex items-center justify-between gap-3">
          <span
            className="truncate font-mono text-xs text-muted-foreground"
            title={info.mlflow_run_id ? `MLflow run ${info.mlflow_run_id}` : undefined}
          >
            v{info.version}
          </span>
          <Link
            href={`/leaderboard?task=${task}`}
            className="inline-flex items-center gap-1 rounded-full px-2 py-1 text-sm font-medium text-violet-700 hover:text-violet-900 dark:text-violet-300 dark:hover:text-violet-200 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-ring/50"
          >
            Leaderboard <ArrowUpRight className="size-3.5" aria-hidden="true" />
          </Link>
        </div>
      </CardContent>
    </Card>
  );
}

function ChipGroup({ title, rows }: { title: string; rows: { group: string; n: number; at_risk_rate: number }[] }) {
  return (
    <div>
      <div className="mb-2 text-xs font-medium tracking-wide text-muted-foreground uppercase">{title}</div>
      <ul className="flex flex-wrap gap-2">
        {rows.map((r) => (
          <li
            key={r.group}
            className="inline-flex items-center gap-2 rounded-full border border-border bg-background/60 py-1.5 pr-2 pl-2.5 text-sm"
          >
            <CheckCircle2 className="size-4 text-lime-600 dark:text-lime-400" aria-hidden="true" />
            <span>{r.group}</span>
            <span className="text-xs text-muted-foreground">n={r.n}</span>
            <span className="ov-gradient-text font-mono text-sm font-semibold tabular-nums">{pct(r.at_risk_rate, 0)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
