import { Skeleton } from "@/components/ui/skeleton";

/** Shown while the next page waits on the API, so navigation responds at once. */
export default function AppLoading() {
  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-8" aria-busy="true" aria-label="Loading page">
      <div className="ep-aurora flex min-h-64 flex-col items-center justify-center gap-4 rounded-[28px] border border-white/10 px-6 py-12 sm:min-h-72">
        <Skeleton className="h-6 w-40 rounded-full bg-white/10" />
        <Skeleton className="h-12 w-72 rounded-xl bg-white/15 sm:w-96" />
        <Skeleton className="h-4 w-80 rounded-full bg-white/10 sm:w-[32rem]" />
      </div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {Array.from({ length: 4 }, (_, i) => (
          <Skeleton key={i} className="ep-tile min-h-40 rounded-3xl" />
        ))}
      </div>
      <div>
        <Skeleton className="ep-tile min-h-80 rounded-3xl" />
      </div>
    </div>
  );
}
