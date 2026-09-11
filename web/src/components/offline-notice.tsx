import { ServerOff } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export function OfflineNotice() {
  return (
    <Card className="glass mx-auto mt-10 max-w-lg">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ServerOff className="size-5 text-destructive" aria-hidden="true" />
          The EduPulse API is not reachable
        </CardTitle>
        <CardDescription>
          Start the backend and reload this page. From the project root:
        </CardDescription>
      </CardHeader>
      <CardContent>
        <pre className="rounded-md bg-muted p-3 font-mono text-xs">
{`edupulse train --all   # once, to register models
edupulse serve         # http://localhost:8000`}
        </pre>
        <p className="mt-3 text-xs text-muted-foreground">
          The frontend reads the backend address from the API_URL environment variable (default http://localhost:8000).
        </p>
      </CardContent>
    </Card>
  );
}
