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
import { OfflineNotice } from "@/components/offline-notice";
import {
  Accent,
  Chip,
  HERO_BUTTON_PRIMARY,
  HERO_BUTTON_SECONDARY,
  PageHero,
  Panel,
  SectionHeading,
  Tile,
} from "@/components/splash";

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
      <PageHero
        eyebrows={[
          health ? `Release ${health.version}` : "Release",
          `${tasks.length} models registered, ${stats.n_students.toLocaleString()} students`,
        ]}
        title={
          <>
            Student performance
            <br />
            intelligence
          </>
        }
        description="The at-risk model flags students likely to average below 60 from background information alone, before any exam is taken. Every prediction comes with a SHAP explanation and every model ships with a fairness audit."
      >
        <Button render={<Link href="/predict" />} nativeButton={false} size="lg" className={HERO_BUTTON_PRIMARY}>
          Score a student <ArrowRight className="size-4" aria-hidden="true" />
        </Button>
        <Button render={<Link href="/leaderboard" />} nativeButton={false} size="lg" className={HERO_BUTTON_SECONDARY}>
          Model leaderboard
        </Button>
      </PageHero>

      <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4" data-reveal="items" aria-label="Key figures">
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

      <section className="flex flex-col gap-6" aria-labelledby="models-heading">
        <SectionHeading
          id="models-heading"
          eyebrow="Registered models"
          title={
            <>
              Three tasks, <Accent>one pipeline</Accent>
            </>
          }
          description="Latest version per task. Hold-out metrics are computed on 200 students the models never saw."
        />
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3" data-reveal="items">
          {tasks.map((t) => (
            <ModelCard key={t} task={t} info={models[t]} />
          ))}
        </div>
      </section>

      <section className="grid grid-cols-1 gap-4 lg:grid-cols-5" aria-labelledby="risk-heading">
        <Panel
          tone="violet"
          className="lg:col-span-2"
          title={<span id="risk-heading">What predicts risk</span>}
          description="At-risk rate by group in the training data."
          contentClassName="flex flex-col gap-5"
        >
          <ChipGroup title="Lunch" rows={lunch} />
          <ChipGroup title="Test preparation" rows={prep} />
          <p className="text-xs text-muted-foreground">
            Free or reduced lunch and no preparation course are the two strongest background signals the at-risk
            model picks up.
          </p>
        </Panel>

        <Panel
          tone="blue"
          className="lg:col-span-3"
          title="At-risk rate by parental education"
          description={`Dashed line is the overall rate (${pct(stats.at_risk_rate)}). Higher parental education tracks lower risk.`}
        >
          <RankedBars
            data={parental.map((g) => ({ name: g.group, rate: g.at_risk_rate, n: g.n }))}
            valueKey="rate"
            format="pct"
            domain={[0, 0.6]}
            reference={{ value: stats.at_risk_rate, label: "overall" }}
          />
        </Panel>
      </section>

      <section
        className="flex flex-col gap-6 rounded-[28px] border border-border bg-card/60 px-6 py-10"
        aria-labelledby="next-heading"
      >
        <SectionHeading
          id="next-heading"
          eyebrow="Explore"
          title={
            <>
              Built for <Accent>accountable</Accent> predictions
            </>
          }
        />
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
                <span className="ep-headline text-base">{title}</span>
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

function ModelCard({ task, info }: { task: TaskName; info: ModelInfo }) {
  const h = headline(info.kind, info.test_metrics);
  return (
    <Panel
      title={
        <span className="flex flex-wrap items-center gap-2">
          {TASK_LABEL[task]}
          {info.leakage_note && (
            <Badge variant="outline" className="text-[10px]">
              leakage case study
            </Badge>
          )}
        </span>
      }
      description={titleCase(info.model)}
      contentClassName="flex flex-col gap-4"
    >
      <dl className="grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-xl bg-foreground/5 p-3">
          <dt className="text-xs text-muted-foreground">Cross-validation {info.cv_metric}</dt>
          <dd className="ep-headline mt-1 text-2xl tabular-nums">{fmt(info.cv_score)}</dd>
        </div>
        <div className="rounded-xl bg-foreground/5 p-3">
          <dt className="text-xs text-muted-foreground">Hold-out {h.label}</dt>
          <dd className="ep-headline mt-1 text-2xl tabular-nums">{h.value}</dd>
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
          className="inline-flex items-center gap-1 rounded-full px-2 py-1 text-sm font-medium text-violet-700 hover:text-violet-900 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-ring/50 dark:text-violet-300 dark:hover:text-violet-200"
        >
          Leaderboard <ArrowUpRight className="size-3.5" aria-hidden="true" />
        </Link>
      </div>
    </Panel>
  );
}

function ChipGroup({ title, rows }: { title: string; rows: { group: string; n: number; at_risk_rate: number }[] }) {
  return (
    <div>
      <div className="mb-2 text-xs font-medium tracking-wide text-muted-foreground uppercase">{title}</div>
      <ul className="flex flex-wrap gap-2">
        {rows.map((r) => (
          <li key={r.group}>
            <Chip icon={CheckCircle2} value={pct(r.at_risk_rate, 0)}>
              {r.group} <span className="text-xs text-muted-foreground">n={r.n}</span>
            </Chip>
          </li>
        ))}
      </ul>
    </div>
  );
}
