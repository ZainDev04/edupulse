# EduPulse web frontend

Next.js 16 (App Router) + Tailwind 4 + shadcn/ui + Recharts. Talks to the FastAPI backend in the parent
repository. Server components fetch from `API_URL` directly; the Predict form goes through the `/api/*`
proxy route so the browser never needs the backend address.

```bash
npm install
API_URL=http://localhost:8000 npm run dev     # http://localhost:3000
npm run build && npm start                    # production
```

Pages: `/` overview, `/predict`, `/leaderboard`, `/explain`, `/fairness`. Each data page takes `?task=`
(`at_risk`, `math_score`, `performance_level`).

Design tokens come from `../design-system/edupulse/MASTER.md`. The stat card and radial gauge were adapted
from 21st.dev components (credited in the source files).
