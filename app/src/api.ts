// Talks to the local engine server (backend/tamo_server). Same origin when the server
// serves the built interface; VITE_API_BASE points elsewhere, e.g. from the packaged app.
const BASE = import.meta.env.VITE_API_BASE ?? "";

export type View = "front" | "three_quarter" | "side" | "back" | "look" | "unsorted";
export const VIEWS: { value: View; label: string }[] = [
  { value: "front", label: "Front" },
  { value: "three_quarter", label: "Three-quarter" },
  { value: "side", label: "Side" },
  { value: "back", label: "Back" },
  { value: "look", label: "Final look" },
  { value: "unsorted", label: "Unsorted" },
];

export type Project = { id: number; slug: string; name: string; created: string; actor_count: number };

export type ActorSummary = {
  id: number;
  project_id: number;
  slug: string;
  name: string;
  preferred_generation_id: number | null;
  front_reference_id: number | null;
  generation_count: number;
};

export type Reference = {
  id: number;
  actor_id: number;
  view: View;
  path: string;
  windows_path: string | null;
  width: number;
  height: number;
  created: string;
};

export type Model = "trellis2" | "pixal3d";

export type Settings = { texture: boolean; pipeline: string; seed: number; smooth: number; crop: boolean };

// What a job was started with: the settings plus the model, and for multi-view its turned images.
export type JobSettings = Partial<Settings> & {
  model?: Model;
  side_id?: number | null;
  three_quarter_id?: number | null;
};

export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled";

export type Job = {
  id: number;
  actor_id: number;
  actor_name?: string;
  reference_id: number;
  settings: JobSettings;
  status: JobStatus;
  stage: string | null;
  progress: number;
  error: string | null;
  run_dir: string | null;
  attempts: number;
  created: string;
  started: string | null;
  finished: string | null;
};

export type RunRecord = {
  model?: Model;
  source_image?: string;
  mode?: string;
  pipeline?: string;
  seed?: number;
  smooth?: number;
  started?: string;
  seconds?: number;
  vram_peak_mib?: number;
  vram_peak_above_baseline_mib?: number;
  cropped_to_head_and_neck?: boolean;
  result?: string;
};

export type Generation = {
  id: number;
  actor_id: number;
  job_id: number | null;
  run_dir: string;
  windows_path: string | null;
  record: RunRecord;
  files: string[];
  created: string;
};

export type Actor = ActorSummary & {
  project_slug: string;
  references: Reference[];
  generations: Generation[];
  jobs: Job[];
};

export type Health = {
  ok: boolean;
  engine: string;
  engine_found: boolean;
  multiview_found?: boolean;
  library: string;
  library_windows_path: string | null;
  trash_windows_path: string | null;
  commit: string | null;
  date: string | null;
  shell_version: string | null;
};

export type UpdateCheck = { commit: string | null; behind: number; changes: string[] };

export class ApiError extends Error {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(BASE + path, init);
  } catch {
    throw new ApiError("Can't reach the engine server. Is it running in Ubuntu?");
  }
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch {
      // not JSON; keep the status line
    }
    throw new ApiError(message);
  }
  return response.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

const form = (fields: Record<string, string | Blob>, method = "POST"): RequestInit => {
  const data = new FormData();
  for (const [key, value] of Object.entries(fields)) data.append(key, value);
  return { method, body: data };
};

export const api = {
  health: () => request<Health>("/api/health"),
  projects: () => request<Project[]>("/api/projects"),
  createProject: (name: string) => request<Project>("/api/projects", json({ name })),
  actors: (projectId: number) => request<ActorSummary[]>(`/api/projects/${projectId}/actors`),
  createActor: (projectId: number, name: string) =>
    request<ActorSummary>(`/api/projects/${projectId}/actors`, json({ name })),
  actor: (actorId: number) => request<Actor>(`/api/actors/${actorId}`),
  uploadReference: (actorId: number, file: File, view: View) =>
    request<Reference>(`/api/actors/${actorId}/references`, form({ file, view })),
  setReferenceView: (referenceId: number, view: View) =>
    request<Reference>(`/api/references/${referenceId}`, form({ view }, "PATCH")),
  generate: (actorId: number, referenceId: number, settings: JobSettings) =>
    request<Job>(`/api/actors/${actorId}/generations`, json({ reference_id: referenceId, ...settings })),
  jobs: (active = false) => request<Job[]>(`/api/jobs${active ? "?active=true" : ""}`),
  cancelJob: (jobId: number) => request<Job>(`/api/jobs/${jobId}/cancel`, { method: "POST" }),
  setPreferred: (actorId: number, generationId: number) =>
    request<ActorSummary>(`/api/actors/${actorId}/preferred`, form({ generation_id: String(generationId) })),
  // Deleting moves the files to the library's trash folder; nothing is erased.
  deleteProject: (projectId: number) => request(`/api/projects/${projectId}`, { method: "DELETE" }),
  deleteActor: (actorId: number) => request(`/api/actors/${actorId}`, { method: "DELETE" }),
  deleteReference: (referenceId: number) => request(`/api/references/${referenceId}`, { method: "DELETE" }),
  deleteGeneration: (generationId: number) => request(`/api/generations/${generationId}`, { method: "DELETE" }),
  checkUpdate: () => request<UpdateCheck>("/api/update/check", { method: "POST" }),
  update: () => request<{ from: string | null; commit: string | null }>("/api/update", { method: "POST" }),
  rescan: () => request<{ added_generations: number }>("/api/rescan", { method: "POST" }),
};

export const referenceImageUrl = (referenceId: number) => `${BASE}/api/references/${referenceId}/image`;
export const generationFileUrl = (generationId: number, name: string) =>
  `${BASE}/api/generations/${generationId}/files/${name}`;

export const DEFAULT_SETTINGS: Settings = {
  texture: true,
  pipeline: "1024_cascade",
  seed: 42,
  smooth: 1,
  crop: true,
};
