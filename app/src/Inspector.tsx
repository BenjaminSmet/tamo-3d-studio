import { useEffect, useState } from "react";
import { api, DEFAULT_SETTINGS, generationFileUrl, VIEWS, type Actor, type Settings, type View } from "./api";
import type { Selection } from "./App";
import { shownGeneration } from "./Studio";
import { Button, ConfirmDelete, Field, Section, formatDate, formatSeconds } from "./ui";

type Props = {
  actor: Actor | null;
  selection: Selection;
  onSelect: (selection: Selection) => void;
  onChanged: () => void;
  onError: (e: unknown) => void;
};

const FILE_LABELS: Record<string, string> = {
  "head_metahuman.fbx": "MetaHuman head (FBX)",
  "model.glb": "Raw model (GLB)",
  "model.obj": "Raw model (OBJ)",
  "input_rgba.png": "What TRELLIS.2 saw",
};

function CopyPath({ path }: { path: string | null }) {
  const [copied, setCopied] = useState(false);
  if (!path) return null;
  return (
    <div className="mt-2">
      <p className="mb-1 text-xs text-neutral-500">Folder (paste into Explorer)</p>
      <div className="flex gap-2">
        <code className="min-w-0 flex-1 truncate rounded bg-[#141517] px-2 py-1 text-[11px]" title={path}>
          {path}
        </code>
        <Button
          onClick={() =>
            navigator.clipboard.writeText(path).then(() => {
              setCopied(true);
              setTimeout(() => setCopied(false), 1500);
            })
          }
        >
          {copied ? "Copied" : "Copy"}
        </Button>
      </div>
    </div>
  );
}

function GenerateForm({ actor, onChanged, onError }: { actor: Actor; onChanged: () => void; onError: (e: unknown) => void }) {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [referenceId, setReferenceId] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const fronts = actor.references.filter((r) => r.view === "front" || r.view === "unsorted");

  useEffect(() => {
    setReferenceId(fronts[0]?.id ?? null);
    // Reset when switching actors or when a new front image arrives.
  }, [actor.id, fronts[0]?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const set = <K extends keyof Settings>(key: K, value: Settings[K]) => setSettings((s) => ({ ...s, [key]: value }));

  return (
    <Section title="Generate">
      {fronts.length === 0 ? (
        <p className="text-neutral-500">Add a front image first. TRELLIS.2 builds the head from one front photo.</p>
      ) : (
        <>
          <Field label="Front image">
            <select
              value={referenceId ?? ""}
              onChange={(e) => setReferenceId(Number(e.target.value))}
              className="max-w-40 rounded border border-edge bg-[#141517] px-1 py-0.5"
            >
              {fronts.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.path.split("/").pop()}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Texture">
            <input type="checkbox" checked={settings.texture} onChange={(e) => set("texture", e.target.checked)} />
          </Field>
          <Field label="Crop to head and neck">
            <input type="checkbox" checked={settings.crop} onChange={(e) => set("crop", e.target.checked)} />
          </Field>
          <Field label="Resolution">
            <select
              value={settings.pipeline}
              onChange={(e) => set("pipeline", e.target.value)}
              className="rounded border border-edge bg-[#141517] px-1 py-0.5"
            >
              <option value="1024_cascade">1024 (recommended)</option>
              <option value="1536_cascade">1536 (panels on skin)</option>
              <option value="1024">1024, single pass</option>
              <option value="512">512, quick test</option>
            </select>
          </Field>
          <Field label={`Smoothing ${settings.smooth.toFixed(1)}`}>
            <input
              type="range"
              min={0}
              max={2}
              step={0.5}
              value={settings.smooth}
              onChange={(e) => set("smooth", Number(e.target.value))}
            />
          </Field>
          <Field label="Seed">
            <input
              type="number"
              value={settings.seed}
              onChange={(e) => set("seed", Number(e.target.value))}
              className="w-20 rounded border border-edge bg-[#141517] px-1 py-0.5 text-right"
            />
          </Field>
          <Button
            variant="primary"
            className="mt-2 w-full"
            disabled={busy || referenceId == null}
            onClick={async () => {
              setBusy(true);
              try {
                await api.generate(actor.id, referenceId!, settings);
                onChanged();
              } catch (e) {
                onError(e);
              } finally {
                setBusy(false);
              }
            }}
          >
            Generate head
          </Button>
          <p className="mt-2 text-xs text-neutral-500">Keep Unreal closed while it runs; they share the GPU.</p>
        </>
      )}
    </Section>
  );
}

export function Inspector({ actor, selection, onSelect, onChanged, onError }: Props) {
  if (!actor) return <aside className="w-80 shrink-0 border-l border-edge bg-panel" />;

  const reference = selection?.kind === "reference" ? actor.references.find((r) => r.id === selection.id) : undefined;
  const generation = reference ? undefined : shownGeneration(actor, selection);
  const job = generation?.job_id ? actor.jobs.find((j) => j.id === generation.job_id) : undefined;

  return (
    <aside className="flex w-80 shrink-0 flex-col overflow-y-auto border-l border-edge bg-panel">
      <GenerateForm actor={actor} onChanged={onChanged} onError={onError} />

      {reference && (
        <Section title="Image">
          <Field label="View">
            <select
              value={reference.view}
              onChange={(e) =>
                api.setReferenceView(reference.id, e.target.value as View).then(onChanged).catch(onError)
              }
              className="rounded border border-edge bg-[#141517] px-1 py-0.5"
            >
              {VIEWS.map((v) => (
                <option key={v.value} value={v.value}>
                  {v.label}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Size">
            <span>
              {reference.width} × {reference.height}
            </span>
          </Field>
          <Field label="Added">
            <span>{formatDate(reference.created)}</span>
          </Field>
          {reference.view !== "front" && (
            <p className="mt-2 text-xs text-neutral-500">
              Side and three-quarter views are stored for multi-view generation later; today only the front image is
              used.
            </p>
          )}
          <CopyPath path={reference.windows_path} />
          <div className="mt-3 flex justify-end">
            <ConfirmDelete
              label="Delete image"
              onConfirm={() =>
                api
                  .deleteReference(reference.id)
                  .then(() => {
                    onSelect(null);
                    onChanged();
                  })
                  .catch(onError)
              }
            />
          </div>
        </Section>
      )}

      {generation && (
        <Section
          title="Version"
          action={
            actor.preferred_generation_id === generation.id ? (
              <span className="text-xs text-amber-400">★ Preferred</span>
            ) : (
              <button
                className="text-xs text-neutral-400 hover:text-amber-300"
                onClick={() =>
                  api
                    .setPreferred(actor.id, generation.id)
                    .then(() => {
                      onChanged();
                      onSelect({ kind: "generation", id: generation.id });
                    })
                    .catch(onError)
                }
              >
                Mark preferred
              </button>
            )
          }
        >
          <Field label="Result">
            <span className={generation.record.result === "ok" ? "text-emerald-400" : "text-red-400"}>
              {generation.record.result ?? "unknown"}
            </span>
          </Field>
          <Field label="Made">
            <span>{formatDate(generation.record.started ?? generation.created)}</span>
          </Field>
          <Field label="Resolution">
            <span>{generation.record.pipeline ?? "–"}</span>
          </Field>
          <Field label="Mode">
            <span>{generation.record.mode ?? "–"}</span>
          </Field>
          <Field label="Seed">
            <span>{generation.record.seed ?? "–"}</span>
          </Field>
          <Field label="Smoothing">
            <span>{generation.record.smooth ?? "–"}</span>
          </Field>
          <Field label="Time">
            <span>{formatSeconds(generation.record.seconds)}</span>
          </Field>
          <Field label="Peak GPU memory">
            <span>
              {generation.record.vram_peak_mib != null ? `${(generation.record.vram_peak_mib / 1024).toFixed(1)} GB` : "–"}
            </span>
          </Field>

          <div className="mt-3 space-y-1">
            {Object.entries(FILE_LABELS)
              .filter(([name]) => generation.files.includes(name))
              .map(([name, label]) => (
                <a
                  key={name}
                  href={generationFileUrl(generation.id, name)}
                  download={name}
                  className={`block rounded px-3 py-1.5 text-center ${
                    name === "head_metahuman.fbx"
                      ? "bg-accent text-white hover:bg-blue-500"
                      : "bg-panel-raised hover:bg-edge"
                  }`}
                >
                  {label}
                </a>
              ))}
            {!generation.files.includes("head_metahuman.fbx") && generation.record.result === "ok" && (
              <p className="text-xs text-neutral-500">
                No MetaHuman FBX in this run (older run, or Blender wasn't found).
              </p>
            )}
          </div>
          {generation.files.includes("head_metahuman.fbx") && (
            <p className="mt-2 text-xs text-neutral-500">
              In Unreal: import with default settings, then MetaHuman Character → Import → From Custom Mesh.
            </p>
          )}
          {job?.error && (
            <pre className="mt-3 max-h-40 overflow-auto whitespace-pre-wrap rounded bg-[#141517] p-2 text-[11px] text-red-300">
              {job.error}
            </pre>
          )}
          <CopyPath path={generation.windows_path} />
          <div className="mt-3 flex justify-end">
            <ConfirmDelete
              label="Delete version"
              onConfirm={() =>
                api
                  .deleteGeneration(generation.id)
                  .then(() => {
                    onSelect(null);
                    onChanged();
                  })
                  .catch(onError)
              }
            />
          </div>
        </Section>
      )}
    </aside>
  );
}
