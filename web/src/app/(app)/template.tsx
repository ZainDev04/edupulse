import { AppReveal } from "@/components/app-reveal";

/**
 * Re-mounted on every navigation inside the app group, so the page enter
 * animation (.ep-page in globals.css) plays each time and the section
 * reveal starts fresh.
 */
export default function AppTemplate({ children }: { children: React.ReactNode }) {
  return <AppReveal>{children}</AppReveal>;
}
