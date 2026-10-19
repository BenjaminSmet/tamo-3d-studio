import { useEffect, useState } from "react";
import { api, type Health, type UpdateCheck } from "./api";

// The Windows app opens the interface with ?shell=<its version>; remember it across reloads.
function installedShell(): string | null {
  const fromUrl = new URLSearchParams(location.search).get("shell");
  try {
    if (fromUrl) sessionStorage.setItem("shell", fromUrl);
    return fromUrl ?? sessionStorage.getItem("shell");
  } catch {
    return fromUrl;
  }
}

const INSTALLER_PAGE = "https://github.com/BenjaminSmet/tamo-3d-studio/actions/workflows/windows-app.yml";

type State =
  | { kind: "idle" }
  | { kind: "checking" }
  | { kind: "current" }
  | { kind: "available"; check: UpdateCheck }
  | { kind: "updating"; from: string | null }
  | { kind: "error"; message: string };

// "Check for updates" in the header: pulls the latest version from GitHub, restarts the
// engine with it, and reloads once the engine answers with the new version.
export function Updates({ health }: { health: Health | null }) {
  const [state, setState] = useState<State>({ kind: "idle" });
  const shell = installedShell();

  useEffect(() => {
    if (state.kind !== "updating") return;
    const started = Date.now();
    let wentDown = false;
    const timer = setInterval(async () => {
      try {
        const now = await api.health();
        if (wentDown || (state.from && now.commit !== state.from)) location.reload();
      } catch {
        wentDown = true;   // restarting; the interface may be rebuilding
      }
      if (Date.now() - started > 5 * 60_000) {
        clearInterval(timer);
        setState({
          kind: "error",
          message: "The engine didn't come back. In Ubuntu, run cd ~/tamo-3d-studio && ./studio.sh to see why.",
        });
      }
    }, 2000);
    return () => clearInterval(timer);
  }, [state]);

  const check = async () => {
    setState({ kind: "checking" });
    try {
      const result = await api.checkUpdate();
      setState(result.behind ? { kind: "available", check: result } : { kind: "current" });
    } catch (e) {
      setState({ kind: "error", message: e instanceof Error ? e.message : String(e) });
    }
  };

  const install = async () => {
    const from = health?.commit ?? null;
    try {
      await api.update();
      setState({ kind: "updating", from });
    } catch (e) {
      setState({ kind: "error", message: e instanceof Error ? e.message : String(e) });
    }
  };

  const link = "text-neutral-400 hover:text-white";
  return (
    <>
      {shell && health?.shell_version && shell !== health.shell_version && (
        <a href={INSTALLER_PAGE} target="_blank" rel="noreferrer" className="text-amber-300 hover:underline">
          New Windows app ({health.shell_version}) available: download and install it
        </a>
      )}
      {state.kind === "idle" && (
        <button className={link} onClick={check} title={health?.commit ? `Version ${health.commit}` : undefined}>
          Check for updates
        </button>
      )}
      {state.kind === "checking" && <span>Checking…</span>}
      {state.kind === "current" && (
        <button className={link} onClick={() => setState({ kind: "idle" })}>
          Up to date
        </button>
      )}
      {state.kind === "available" && (
        <button
          className="rounded bg-accent px-2 py-0.5 text-white hover:bg-blue-500"
          title={state.check.changes.join("\n")}
          onClick={install}
        >
          Update ({state.check.behind} {state.check.behind === 1 ? "change" : "changes"})
        </button>
      )}
      {state.kind === "updating" && <span className="text-accent">Updating… the app reloads by itself</span>}
      {state.kind === "error" && (
        <button className="max-w-md truncate text-red-300" title={state.message} onClick={() => setState({ kind: "idle" })}>
          Update failed: {state.message}
        </button>
      )}
    </>
  );
}
