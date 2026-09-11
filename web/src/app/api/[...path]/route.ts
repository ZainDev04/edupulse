/**
 * Thin proxy so client components can call the FastAPI backend from the
 * browser without CORS or exposing its address. Forwards the path, query
 * string, method, JSON body and status code unchanged.
 */
import { API_URL } from "@/lib/api";

export const dynamic = "force-dynamic";

async function forward(req: Request, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const incoming = new URL(req.url);
  const target = `${API_URL}/${path.join("/")}${incoming.search}`;
  const init: RequestInit = {
    method: req.method,
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
  };
  if (req.method !== "GET" && req.method !== "HEAD") {
    init.body = await req.text();
  }
  try {
    const res = await fetch(target, init);
    const body = await res.text();
    return new Response(body, {
      status: res.status,
      headers: { "Content-Type": res.headers.get("Content-Type") ?? "application/json" },
    });
  } catch {
    return Response.json({ detail: `Backend unreachable at ${API_URL}` }, { status: 503 });
  }
}

export { forward as GET, forward as POST };
