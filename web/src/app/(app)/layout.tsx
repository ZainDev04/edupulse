import { AppShell } from "@/components/app-shell";
import { AppTheme, THEME_SCRIPT } from "@/components/theme";
import { api } from "@/lib/api";

export default async function AppLayout({ children }: { children: React.ReactNode }) {
  const health = await api.health();
  return (
    <>
      {/* Applies the stored theme (light by default) before hydration */}
      <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      <AppTheme />
      <AppShell health={health}>{children}</AppShell>
    </>
  );
}
