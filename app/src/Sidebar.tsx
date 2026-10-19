import { useState } from "react";
import { api, referenceImageUrl, type ActorSummary, type Job, type Project } from "./api";
import { Button, ConfirmDelete, ProgressBar, Section } from "./ui";

type Props = {
  projects: Project[];
  projectId: number | null;
  onProject: (id: number) => void;
  onProjectCreated: (project: Project) => void;
  actors: ActorSummary[];
  actorId: number | null;
  onActor: (id: number) => void;
  onActorCreated: (actor: ActorSummary) => void;
  onProjectDeleted: () => void;
  jobs: Job[];
  onError: (e: unknown) => void;
};

function NameForm({ placeholder, onSubmit }: { placeholder: string; onSubmit: (name: string) => Promise<void> }) {
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  return (
    <form
      className="mt-2 flex gap-2"
      onSubmit={async (event) => {
        event.preventDefault();
        if (!name.trim()) return;
        setBusy(true);
        try {
          await onSubmit(name.trim());
          setName("");
        } finally {
          setBusy(false);
        }
      }}
    >
      <input
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder={placeholder}
        className="min-w-0 flex-1 rounded border border-edge bg-[#141517] px-2 py-1 outline-none focus:border-accent"
      />
      <Button type="submit" disabled={busy || !name.trim()}>
        Add
      </Button>
    </form>
  );
}

export function Sidebar(props: Props) {
  const { projects, projectId, actors, actorId, jobs, onError } = props;
  const [addingProject, setAddingProject] = useState(false);

  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-edge bg-panel">
      <Section
        title="Project"
        action={
          <button className="text-xs text-neutral-400 hover:text-white" onClick={() => setAddingProject((v) => !v)}>
            {addingProject ? "Close" : "New"}
          </button>
        }
      >
        <select
          value={projectId ?? ""}
          onChange={(e) => props.onProject(Number(e.target.value))}
          className="w-full rounded border border-edge bg-[#141517] px-2 py-1.5"
        >
          {projects.length === 0 && <option value="">No projects yet</option>}
          {projects.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
        {projectId != null && !addingProject && (
          <div className="mt-2 flex justify-end">
            <ConfirmDelete
              label="Delete project"
              onConfirm={() => api.deleteProject(projectId).then(props.onProjectDeleted).catch(onError)}
            />
          </div>
        )}
        {(addingProject || projects.length === 0) && (
          <NameForm
            placeholder="Project name"
            onSubmit={async (name) => {
              try {
                props.onProjectCreated(await api.createProject(name));
                setAddingProject(false);
              } catch (e) {
                onError(e);
              }
            }}
          />
        )}
      </Section>

      <Section title="Actors">
        <ul className="-mx-1 max-h-[45vh] overflow-y-auto">
          {actors.map((a) => (
            <li key={a.id}>
              <button
                onClick={() => props.onActor(a.id)}
                className={`flex w-full items-center gap-2 rounded px-1 py-1 text-left ${
                  a.id === actorId ? "bg-accent/20 text-white" : "hover:bg-panel-raised"
                }`}
              >
                {a.front_reference_id ? (
                  <img
                    src={referenceImageUrl(a.front_reference_id)}
                    alt=""
                    className="h-8 w-8 rounded object-cover"
                  />
                ) : (
                  <span className="h-8 w-8 rounded bg-panel-raised" />
                )}
                <span className="min-w-0 flex-1 truncate">{a.name}</span>
                <span className="text-xs text-neutral-500">{a.generation_count || ""}</span>
              </button>
            </li>
          ))}
        </ul>
        {projectId != null && (
          <NameForm
            placeholder="Actor name"
            onSubmit={async (name) => {
              try {
                props.onActorCreated(await api.createActor(projectId, name));
              } catch (e) {
                onError(e);
              }
            }}
          />
        )}
      </Section>

      <div className="mt-auto">
        <Section title={`Queue${jobs.length ? ` (${jobs.length})` : ""}`}>
          {jobs.length === 0 && <p className="text-neutral-500">Nothing running.</p>}
          <ul className="space-y-3">
            {jobs.map((job) => (
              <li key={job.id}>
                <div className="mb-1 flex items-center justify-between gap-2">
                  <span className="truncate">{job.actor_name}</span>
                  <button
                    className="text-xs text-neutral-400 hover:text-red-300"
                    onClick={() => api.cancelJob(job.id).catch(onError)}
                  >
                    Cancel
                  </button>
                </div>
                <ProgressBar value={job.progress} />
                <p className="mt-1 text-xs text-neutral-500">
                  {job.status === "queued" ? "Waiting for the GPU" : job.stage}
                </p>
              </li>
            ))}
          </ul>
        </Section>
      </div>
    </aside>
  );
}
