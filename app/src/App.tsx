import { useCallback, useEffect, useRef, useState } from "react";
import { api, type Actor, type ActorSummary, type Health, type Job, type Project } from "./api";
import { Inspector } from "./Inspector";
import { Sidebar } from "./Sidebar";
import { Studio } from "./Studio";
import { Updates } from "./Updates";

export type Selection = { kind: "reference"; id: number } | { kind: "generation"; id: number } | null;

const POLL_MS = 1500;

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState<number | null>(null);
  const [actors, setActors] = useState<ActorSummary[]>([]);
  const [actorId, setActorId] = useState<number | null>(null);
  const [actor, setActor] = useState<Actor | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selection, setSelection] = useState<Selection>(null);
  const [error, setError] = useState<string | null>(null);

  const report = useCallback((e: unknown) => setError(e instanceof Error ? e.message : String(e)), []);

  const loadProjects = useCallback(async () => {
    const list = await api.projects();
    setProjects(list);
    setProjectId((current) => current ?? list[0]?.id ?? null);
  }, []);

  const loadActors = useCallback(async () => {
    if (projectId == null) return setActors([]);
    setActors(await api.actors(projectId));
  }, [projectId]);

  const loadActor = useCallback(async () => {
    if (actorId == null) {
      setActor(null);
      return null;
    }
    const loaded = await api.actor(actorId);
    setActor(loaded);
    return loaded;
  }, [actorId]);

  useEffect(() => {
    api.health().then(setHealth).catch(report);
    loadProjects().catch(report);
  }, [loadProjects, report]);

  useEffect(() => {
    setActorId(null);
    loadActors().catch(report);
  }, [loadActors, report]);

  useEffect(() => {
    setSelection(null);
    loadActor().catch(report);
  }, [loadActor, report]);

  // Poll the queue; when a job finishes, refresh what it changed and show its result.
  const activeIds = useRef<string>("");
  const previous = useRef<Job[]>([]);
  useEffect(() => {
    let stopped = false;
    const tick = async () => {
      try {
        const active = await api.jobs(true);
        if (stopped) return;
        setJobs(active);
        const ids = active.map((j) => j.id).join(",");
        if (ids !== activeIds.current) {
          activeIds.current = ids;
          const finished = previous.current.filter((j) => !active.some((a) => a.id === j.id)).map((j) => j.id);
          const [loaded] = await Promise.all([loadActor(), loadActors()]);
          const result = loaded?.generations.find((g) => g.job_id != null && finished.includes(g.job_id));
          if (result && !stopped) setSelection({ kind: "generation", id: result.id });
        }
        previous.current = active;
        setError((current) => (current?.startsWith("Can't reach") ? null : current));
      } catch (e) {
        if (!stopped) report(e);
      }
    };
    tick();
    const timer = setInterval(tick, POLL_MS);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, [loadActor, loadActors, report]);

  const refreshActor = useCallback(() => {
    loadActor().catch(report);
    loadActors().catch(report);
  }, [loadActor, loadActors, report]);

  return (
    <div className="flex h-full flex-col">
      <header className="flex h-10 shrink-0 items-center gap-3 border-b border-edge bg-panel px-4">
        <span className="font-semibold tracking-wide">Tamo 3D Studio</span>
        <span className="text-neutral-500">Face Studio</span>
        <span className="ml-auto flex items-center gap-4 text-xs text-neutral-400">
          <Updates health={health} />
          <span className="flex items-center gap-2">
            <span
              className={`h-2 w-2 rounded-full ${
                health == null ? "bg-neutral-600" : health.engine_found ? "bg-emerald-500" : "bg-amber-500"
              }`}
            />
            {health == null ? "Connecting…" : health.engine_found ? "Engine ready" : "Engine not found"}
          </span>
        </span>
      </header>

      {error && (
        <div className="flex items-center gap-3 border-b border-red-900 bg-red-950/60 px-4 py-2 text-red-200">
          <span>{error}</span>
          <button className="ml-auto text-red-300 hover:text-white" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      )}

      <div className="flex min-h-0 flex-1">
        <Sidebar
          projects={projects}
          projectId={projectId}
          onProject={setProjectId}
          onProjectCreated={(p) => loadProjects().then(() => setProjectId(p.id)).catch(report)}
          actors={actors}
          actorId={actorId}
          onActor={setActorId}
          onActorCreated={(a) => loadActors().then(() => setActorId(a.id)).catch(report)}
          onProjectDeleted={() => {
            setProjectId(null);
            loadProjects().catch(report);
          }}
          jobs={jobs}
          onError={report}
        />
        <main className="min-w-0 flex-1 bg-[#141517]">
          {actor ? (
            <Studio
              actor={actor}
              jobs={jobs.filter((j) => j.actor_id === actor.id)}
              selection={selection}
              onSelect={setSelection}
              onChanged={refreshActor}
              onDeleted={() => {
                setActorId(null);
                loadActors().catch(report);
              }}
              onError={report}
            />
          ) : (
            <div className="flex h-full items-center justify-center text-neutral-500">
              {projectId == null ? "Create a project to start." : "Pick or add an actor on the left."}
            </div>
          )}
        </main>
        <Inspector
          actor={actor}
          selection={selection}
          onSelect={setSelection}
          onChanged={refreshActor}
          onError={report}
        />
      </div>
    </div>
  );
}
