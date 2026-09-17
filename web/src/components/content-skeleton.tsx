import { Skeleton } from "@/components/ui/skeleton";

/** Placeholder for the content below a task page's hero while the API answers. */
export function ContentSkeleton() {
  return (
    <div className="flex flex-col gap-8" aria-busy="true" aria-label="Loading content">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {Array.from({ length: 4 }, (_, i) => (
          <Skeleton key={i} className="ep-tile min-h-40 rounded-3xl" />
        ))}
      </div>
      <Skeleton className="ep-tile min-h-80 rounded-3xl" />
    </div>
  );
}
