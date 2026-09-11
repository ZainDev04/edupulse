// Adapted from 21st.dev "Progress" (radial variant) by sean0205.
// Pure SVG, no dependencies; the threshold marker was added for EduPulse.
import { cn } from "cn";

export function ProgressRadial({
  className,
  value = 0,
  size = 200,
  strokeWidth = 14,
  startAngle = -210,
  endAngle = 30,
  threshold,
  indicatorClassName,
  trackClassName,
  children,
  label,
}: {
  className?: string;
  value?: number;
  size?: number;
  strokeWidth?: number;
  startAngle?: number;
  endAngle?: number;
  /** Optional 0-100 marker drawn on the arc (the decision threshold). */
  threshold?: number;
  indicatorClassName?: string;
  trackClassName?: string;
  children?: React.ReactNode;
  label: string;
}) {
  const clamped = Math.max(0, Math.min(100, value));
  const radius = (size - strokeWidth) / 2;
  const angleRange = endAngle - startAngle;
  const progressAngle = (clamped / 100) * angleRange;
  const rad = (deg: number) => (deg * Math.PI) / 180;
  const point = (deg: number) => [
    size / 2 + radius * Math.cos(rad(deg)),
    size / 2 + radius * Math.sin(rad(deg)),
  ];
  const arc = (from: number, to: number) => {
    const [sx, sy] = point(from);
    const [ex, ey] = point(to);
    const large = to - from > 180 ? 1 : 0;
    return `M ${sx} ${sy} A ${radius} ${radius} 0 ${large} 1 ${ex} ${ey}`;
  };
  const marker = threshold === undefined ? null : point(startAngle + (threshold / 100) * angleRange);
  const markerInner =
    threshold === undefined
      ? null
      : [
          size / 2 + (radius - strokeWidth) * Math.cos(rad(startAngle + (threshold / 100) * angleRange)),
          size / 2 + (radius - strokeWidth) * Math.sin(rad(startAngle + (threshold / 100) * angleRange)),
        ];

  return (
    <div
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(clamped)}
      className={cn("relative inline-flex items-center justify-center", className)}
      style={{ width: size, height: size }}
    >
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
        <path
          d={arc(startAngle, endAngle)}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="none"
          strokeLinecap="round"
          className={cn("text-muted", trackClassName)}
        />
        {clamped > 0 && (
          <path
            d={arc(startAngle, startAngle + progressAngle)}
            stroke="currentColor"
            strokeWidth={strokeWidth}
            fill="none"
            strokeLinecap="round"
            className={cn("text-primary transition-all duration-500 ease-out", indicatorClassName)}
          />
        )}
        {marker && markerInner && (
          <line
            x1={marker[0]}
            y1={marker[1]}
            x2={markerInner[0]}
            y2={markerInner[1]}
            stroke="currentColor"
            strokeWidth={3}
            strokeLinecap="round"
            className="text-foreground"
          />
        )}
      </svg>
      {children && <div className="absolute inset-0 flex items-center justify-center">{children}</div>}
    </div>
  );
}
