/**
 * Re-mounted on every navigation inside the app group, so the enter
 * animation in globals.css (.ep-page) plays each time a page changes.
 */
export default function AppTemplate({ children }: { children: React.ReactNode }) {
  return <div className="ep-page">{children}</div>;
}
