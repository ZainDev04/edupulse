import type { Metadata } from "next";
import { AlertTriangle, Award, Clock3, Cpu, Target } from "lucide-react";
import { api, fmt, titleCase, type LeaderboardRow } from "@/lib/api";
import { RankedBars } from "@/components/charts/bar-charts";
import { OfflineNotice } from "@/components/offline-notice";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Accent, Chip, Panel, SectionHeading, Tile } from "@/components/splash";
import { pickTask } from "@/lib/tasks";

export const metadata: Metadata = { title: "Leaderboard" };

export default async function LeaderboardPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const task = pickTask((await searchParams).task);
  const [info, board] = await Promise.all([api.model(task), api.leaderboard(task)]);
  if (!info || !board) return <OfflineNotice />;

  const metric = info.primary_metric;
  const meanKey = `${metric}_mean`;
  const stdKey = `${metric}_std`;
  const rows = board.rows as LeaderboardRow[];
  const chart = rows.map((r) => ({
    name: String(r.model),
    value: Number(r[meanKey]),
    std: Number(r[stdKey]),
    family: String(r.family),
  }));
  const best = rows[0];
  const min = Math.min(...chart.map((c) => c.value));
  const secondary = Object.keys(rows[0]).filter((k) => k.endsWith("_mean") && k !== meanKey);
  const holdout =
    info.kind === "regression"
      ? { label: "Hold-out R²", value: info.test_metrics.r2 }
      : info.kind === "binary"
        ? { label: "Hold-out ROC-AUC", value: info.test_metrics.roc_auc }
        : { label: "Hold-out accuracy", value: info.test_metrics.accuracy };

  return (
    <>
      {info.leakage_note && (
        <div className="flex justify-center">
          <Chip icon={AlertTriangle} tone="warn">
            <span className="font-medium">Leakage note.</span> {info.leakage_note}
          </Chip>
        </div>
      )}

      <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4" data-reveal="items" aria-label="Selected model">
        <Tile tone="violet" icon={Award} label="Selected model" value={titleCase(info.model)} hint={info.model_class} />
        <Tile tone="blue" icon={Target} label={`CV ${metric}`} value={fmt(info.cv_score)} hint={`± ${fmt(Number(best[stdKey]))} across folds`} />
        <Tile tone="green" icon={Cpu} label={holdout.label} value={fmt(holdout.value)} hint="200 held-out students" />
        <Tile tone="amber" icon={Clock3} label="Fit time" value={`${Number(best.fit_seconds).toFixed(2)}s`} hint="mean per fold, selected model" />
      </section>

      <section className="flex flex-col gap-6" aria-labelledby="cv-heading">
        <SectionHeading
          id="cv-heading"
          eyebrow="Cross-validation"
          title={
            <>
              Ten families, <Accent>one winner</Accent>
            </>
          }
          description={`Mean ${metric} with standard deviation across 10 folds. Colour is the model family.`}
        />
        <Panel tone="violet" title={`Cross-validated ${metric}`} description="Error bars show one standard deviation across folds.">
          <RankedBars
            data={chart}
            valueKey="value"
            errorKey="std"
            colorKey="family"
            domain={[Math.max(0, Math.floor(min * 10) / 10 - 0.1), 1]}
          />
          <div className="mt-3 flex flex-wrap gap-2">
            {Array.from(new Set(chart.map((c) => c.family))).map((f) => (
              <Badge key={f} variant="outline">
                {f}
              </Badge>
            ))}
          </div>
        </Panel>
      </section>

      <Panel tone="blue" title="All metrics" description="Cross-validation means for every metric the task records." contentClassName="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>#</TableHead>
              <TableHead>Model</TableHead>
              <TableHead>Family</TableHead>
              <TableHead className="text-right">{metric}</TableHead>
              {secondary.map((k) => (
                <TableHead key={k} className="text-right">
                  {k.replace("_mean", "").replace("neg_", "")}
                </TableHead>
              ))}
              <TableHead className="text-right">fit (s)</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r) => (
              <TableRow key={String(r.model)}>
                <TableCell className="font-mono text-xs">{r.rank}</TableCell>
                <TableCell className="font-medium">{String(r.model)}</TableCell>
                <TableCell className="text-muted-foreground">{String(r.family)}</TableCell>
                <TableCell className="text-right font-mono text-xs">{fmt(Number(r[meanKey]))}</TableCell>
                {secondary.map((k) => (
                  <TableCell key={k} className="text-right font-mono text-xs">
                    {fmt(Math.abs(Number(r[k])))}
                  </TableCell>
                ))}
                <TableCell className="text-right font-mono text-xs">{Number(r.fit_seconds).toFixed(2)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Panel>
    </>
  );
}
