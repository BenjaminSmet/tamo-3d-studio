import { useRef, useState } from "react";
import { api, generationFileUrl, referenceImageUrl, VIEWS, type Actor, type Generation, type Job, type View } from "./api";
import type { Selection } from "./App";
import { HeadViewer } from "./HeadViewer";
import { ConfirmDelete, ProgressBar, formatDate } from "./ui";

type Props = {
  actor: Actor;
  jobs: Job[];
  selection: Selection;
  onSelect: (selection: Selection) => void;
  onChanged: () => void;
  onDeleted: () => void;
  onError: (e: unknown) => void;
};

const viewLabel = (view: View) => VIEWS.find((v) => v.value === view)?.label ?? view;

// The generation shown in the viewer: the one selected, else the preferred one, else the newest that worked.
export function shownGeneration(actor: Actor, selection: Selection): Generation | undefined {
  if (selection?.kind === "generation") return actor.generations.find((g) => g.id === selection.id);
  return (
    actor.generations.find((g) => g.id === actor.preferred_generation_id) ??
    actor.generations.find((g) => g.record.result === "ok")
  );
}

export function Studio({ actor, jobs, selection, onSelect, onChanged, onDeleted, onError }: Props) {
  const [shading, setShading] = useState<"texture" | "clay">("texture");
  const [uploadView, setUploadView] = useState<View>("front");
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const shown = shownGeneration(actor, selection);
  // The run folder in the URL: ids of deleted versions get reused, and the viewer and browser cache by URL.
  const modelUrl = shown?.files.includes("model.glb")
    ? `${generationFileUrl(shown.id, "model.glb")}?run=${encodeURIComponent(shown.run_dir)}`
    : null;
  const running = jobs.find((j) => j.status === "running") ?? jobs[0];

  const upload = async (files: FileList | null) => {
    if (!files?.length) return;
    setUploading(true);
    try {
      let last = null;
      for (const file of Array.from(files)) last = await api.uploadReference(actor.id, file, uploadView);
      onChanged();
      if (last) onSelect({ kind: "reference", id: last.id });
    } catch (e) {
      onError(e);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-3 border-b border-edge px-4 py-2">
        <h1 className="text-base font-semibold">{actor.name}</h1>
        <span className="text-neutral-500">
          {actor.generations.length} {actor.generations.length === 1 ? "version" : "versions"}
        </span>
        <ConfirmDelete
          label="Delete actor"
          onConfirm={() => api.deleteActor(actor.id).then(onDeleted).catch(onError)}
        />
        <div className="ml-auto flex overflow-hidden rounded border border-edge text-xs">
          {(["texture", "clay"] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => setShading(mode)}
              className={`px-3 py-1 ${shading === mode ? "bg-panel-raised text-white" : "text-neutral-400 hover:text-white"}`}
            >
              {mode === "texture" ? "Textured" : "Clay"}
            </button>
          ))}
        </div>
      </div>

      {/* Reference images, with a drop zone for more. */}
      <div className="flex gap-2 overflow-x-auto border-b border-edge px-4 py-3">
        {actor.references.map((r) => (
          <button
            key={r.id}
            onClick={() => onSelect({ kind: "reference", id: r.id })}
            className={`relative h-24 w-24 shrink-0 overflow-hidden rounded border ${
              selection?.kind === "reference" && selection.id === r.id ? "border-accent" : "border-edge"
            }`}
          >
            <img src={referenceImageUrl(r.id)} alt="" className="h-full w-full object-cover" />
            <span className="absolute inset-x-0 bottom-0 bg-black/60 px-1 py-0.5 text-[11px]">{viewLabel(r.view)}</span>
          </button>
        ))}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            upload(e.dataTransfer.files);
          }}
          className={`flex h-24 w-48 shrink-0 flex-col items-center justify-center gap-1 rounded border border-dashed text-center text-xs ${
            dragging ? "border-accent bg-accent/10" : "border-edge text-neutral-400"
          }`}
        >
          <span>{uploading ? "Uploading…" : "Drop images here"}</span>
          <div className="flex items-center gap-1">
            <select
              value={uploadView}
              onChange={(e) => setUploadView(e.target.value as View)}
              className="rounded border border-edge bg-[#141517] px-1 py-0.5"
            >
              {VIEWS.filter((v) => v.value !== "unsorted").map((v) => (
                <option key={v.value} value={v.value}>
                  {v.label}
                </option>
              ))}
            </select>
            <button className="text-accent hover:underline" onClick={() => fileInput.current?.click()}>
              or browse
            </button>
          </div>
          <input
            ref={fileInput}
            type="file"
            accept="image/png,image/jpeg,image/webp"
            multiple
            hidden
            onChange={(e) => {
              upload(e.target.files);
              e.target.value = "";
            }}
          />
        </div>
      </div>

      <div className="relative min-h-0 flex-1 overflow-hidden">
        <HeadViewer url={modelUrl} shading={shading} turn={shown?.record.model === "pixal3d"} />
        {running && (
          <div className="absolute left-4 right-4 top-4 rounded border border-edge bg-panel/90 p-3 md:left-auto md:w-80">
            <div className="mb-2 flex justify-between text-xs">
              <span>{running.status === "queued" ? "Waiting for the GPU" : running.stage}</span>
              <span className="text-neutral-400">{Math.round(running.progress * 100)}%</span>
            </div>
            <ProgressBar value={running.progress} />
          </div>
        )}
      </div>

      {/* Every run is a version, newest first; nothing is ever overwritten. Wraps so none hide off to the side. */}
      <div className="flex max-h-32 flex-wrap gap-2 overflow-y-auto border-t border-edge px-4 py-2">
        {actor.generations.length === 0 && (
          <span className="py-1 text-neutral-500">No versions yet. Add a front image and press Generate.</span>
        )}
        {actor.generations.map((g) => (
          <button
            key={g.id}
            onClick={() => onSelect({ kind: "generation", id: g.id })}
            title="Show this version"
            className={`shrink-0 rounded border px-3 py-1.5 text-left text-xs ${
              shown?.id === g.id ? "border-accent bg-accent/10" : "border-edge hover:bg-panel-raised"
            }`}
          >
            <div className="flex items-center gap-1">
              {actor.preferred_generation_id === g.id && <span className="text-amber-400">★</span>}
              <span>{formatDate(g.record.started ?? g.created)}</span>
            </div>
            <div className={g.record.result === "ok" ? "text-neutral-500" : "text-red-400"}>
              {g.record.result === "ok"
                ? g.record.model === "gnm"
                  ? `face model${g.record.mode === "textured" ? " · skin" : ""}`
                  : `${g.record.model === "pixal3d" ? "multi-view · " : ""}${g.record.pipeline ?? ""} · ${g.record.mode ?? ""}`
                : "failed"}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
