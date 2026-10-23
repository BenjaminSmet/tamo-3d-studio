import { useEffect, useState } from "react";
import {
  api,
  DEFAULT_SETTINGS,
  generationFileUrl,
  MODEL_NAMES,
  VIEWS,
  type Actor,
  type Model,
  type Reference,
  type Settings,
  type View,
} from "./api";
import type { Selection } from "./App";
import { shownGeneration } from "./Studio";
import { Button, ConfirmDelete, Field, Section, formatDate, formatSeconds } from "./ui";

type Props = {
  actor: Actor | null;
  selection: Selection;
  onSelect: (selection: Selection) => void;
  onChanged: () => void;
  onError: (e: unknown) => void;
  multiviewFound: boolean;
  facemodelFound: boolean;
};

const FILE_LABELS: Record<string, string> = {
  "head_metahuman.fbx": "MetaHuman head (FBX)",
  "albedo.png": "Skin texture (albedo PNG)",
  "model.glb": "Raw model (GLB)",
  "model.obj": "Raw model (OBJ)",
  "input_rgba.png": "What the model saw (front)",
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

// Without the "<view>-<date>-<time>-" the library adds, so images read by their own names.
const fileName = (r: Reference) => r.path.split("/").pop()?.replace(/^[a-z_]+-\d{8}-\d{6}-/, "");

function ImagePicker({ label, value, options, none, onChange }: {
  label: string;
  value: number | null;
  options: Reference[];
  none?: string;
  onChange: (id: number | null) => void;
}) {
  return (
    <Field label={label}>
      <select
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
        title={options.find((r) => r.id === value)?.path.split("/").pop()}
        className="max-w-40 rounded border border-edge bg-[#141517] px-1 py-0.5"
      >
        {none && <option value="">{none}</option>}
        {options.map((r) => (
          <option key={r.id} value={r.id}>
            {fileName(r)}
          </option>
        ))}
      </select>
    </Field>
  );
}

function GenerateForm({ actor, multiviewFound, facemodelFound, onChanged, onError }: {
  actor: Actor;
  multiviewFound: boolean;
  facemodelFound: boolean;
  onChanged: () => void;
  onError: (e: unknown) => void;
}) {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [referenceId, setReferenceId] = useState<number | null>(null);
  const [sideId, setSideId] = useState<number | null>(null);
  const [threeQuarterId, setThreeQuarterId] = useState<number | null>(null);
  const [otherThreeQuarterId, setOtherThreeQuarterId] = useState<number | null>(null);
  const [otherSideId, setOtherSideId] = useState<number | null>(null);
  const [backId, setBackId] = useState<number | null>(null);
  // The face model's skin texture is on by default (it helps placing MetaHuman's markers); the other methods share settings.texture.
  const [skinTexture, setSkinTexture] = useState(true);
  const [model, setModel] = useState<Model>("trellis2");
  const [busy, setBusy] = useState(false);
  const fronts = actor.references.filter((r) => r.view === "front" || r.view === "unsorted");
  // Newest first, so the default is the latest photo of each view.
  const sides = actor.references.filter((r) => r.view === "side").reverse();
  const threeQuarters = actor.references.filter((r) => r.view === "three_quarter").reverse();
  const otherThreeQuarters = actor.references.filter((r) => r.view === "other_three_quarter").reverse();
  const otherSides = actor.references.filter((r) => r.view === "other_side").reverse();
  const backs = actor.references.filter((r) => r.view === "back").reverse();
  const canMultiview = multiviewFound && sides.length + threeQuarters.length > 0;
  const multiview = model === "pixal3d";
  // The face model uses the turned views when there are some, and builds shape only.
  const facemodel = model === "gnm";
  const turned = multiview || facemodel;

  useEffect(() => {
    setReferenceId(fronts[0]?.id ?? null);
    setSideId(sides[0]?.id ?? null);
    setThreeQuarterId(threeQuarters[0]?.id ?? null);
    setOtherThreeQuarterId(otherThreeQuarters[0]?.id ?? null);
    setOtherSideId(otherSides[0]?.id ?? null);
    setBackId(backs[0]?.id ?? null);
    // The face model is the default when it's installed; otherwise front only (multi-view only does better
    // with uncropped turned views from the same shoot).
    setModel(facemodelFound ? "gnm" : "trellis2");
    // Reset when switching actors or when new images arrive.
  }, [actor.id, fronts[0]?.id, sides[0]?.id, threeQuarters[0]?.id, otherThreeQuarters[0]?.id, otherSides[0]?.id, backs[0]?.id, canMultiview, facemodelFound]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (multiview && !settings.pipeline.endsWith("_cascade")) setSettings((s) => ({ ...s, pipeline: "1024_cascade" }));
  }, [multiview, settings.pipeline]);

  const set = <K extends keyof Settings>(key: K, value: Settings[K]) => setSettings((s) => ({ ...s, [key]: value }));

  return (
    <Section title="Generate">
      {fronts.length === 0 ? (
        <p className="text-neutral-500">Add a front image first. Every head is built from a front photo.</p>
      ) : (
        <>
          <Field label="Method">
            <select
              value={model}
              onChange={(e) => setModel(e.target.value as Model)}
              className="max-w-40 rounded border border-edge bg-[#141517] px-1 py-0.5"
            >
              <option value="trellis2">{MODEL_NAMES.trellis2}</option>
              <option value="pixal3d" disabled={!canMultiview}>
                {MODEL_NAMES.pixal3d}
              </option>
              <option value="gnm" disabled={!facemodelFound}>
                {MODEL_NAMES.gnm}
              </option>
            </select>
          </Field>
          {!canMultiview && !facemodel && (
            <p className="mb-1 text-xs text-neutral-500">
              {multiviewFound
                ? "Add a Side or Three-quarter image for multi-view."
                : "Multi-view needs Pixal3D installed (see docs/multiview.md)."}
            </p>
          )}
          <ImagePicker label="Front image" value={referenceId} options={fronts} onChange={setReferenceId} />
          {turned && (
            <>
              <ImagePicker label="Side image" value={sideId} options={sides} none="None" onChange={setSideId} />
              <ImagePicker
                label="Three-quarter image"
                value={threeQuarterId}
                options={threeQuarters}
                none="None"
                onChange={setThreeQuarterId}
              />
            </>
          )}
          {facemodel && (
            <>
              <ImagePicker
                label="Other three-quarter image"
                value={otherThreeQuarterId}
                options={otherThreeQuarters}
                none="None"
                onChange={setOtherThreeQuarterId}
              />
              <ImagePicker
                label="Other side image"
                value={otherSideId}
                options={otherSides}
                none="None"
                onChange={setOtherSideId}
              />
              <ImagePicker label="Back image" value={backId} options={backs} none="None" onChange={setBackId} />
            </>
          )}
          {facemodel ? (
            <Field label="Skin texture">
              <input type="checkbox" checked={skinTexture} onChange={(e) => setSkinTexture(e.target.checked)} />
            </Field>
          ) : (
            <Field label="Texture">
              <input type="checkbox" checked={settings.texture} onChange={(e) => set("texture", e.target.checked)} />
            </Field>
          )}
          {!turned && (
            <Field label="Crop to head and neck">
              <input type="checkbox" checked={settings.crop} onChange={(e) => set("crop", e.target.checked)} />
            </Field>
          )}
          {!facemodel && (
            <Field label="Resolution">
              <select
                value={settings.pipeline}
                onChange={(e) => set("pipeline", e.target.value)}
                className="rounded border border-edge bg-[#141517] px-1 py-0.5"
              >
                <option value="1024_cascade">1024 (recommended)</option>
                <option value="1536_cascade">1536{multiview ? "" : " (panels on skin)"}</option>
                {!multiview && <option value="1024">1024, single pass</option>}
                {!multiview && <option value="512">512, quick test</option>}
              </select>
            </Field>
          )}
          {!facemodel && (
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
          )}
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
            disabled={busy || referenceId == null || (multiview && sideId == null && threeQuarterId == null)}
            onClick={async () => {
              setBusy(true);
              try {
                const { crop, ...common } = settings;
                await api.generate(
                  actor.id,
                  referenceId!,
                  facemodel
                    ? {
                        model,
                        texture: skinTexture,
                        seed: settings.seed,
                        side_id: sideId,
                        three_quarter_id: threeQuarterId,
                        other_three_quarter_id: otherThreeQuarterId,
                        other_side_id: otherSideId,
                        back_id: backId,
                      }
                    : multiview
                      ? { ...common, model, side_id: sideId, three_quarter_id: threeQuarterId }
                      : { ...common, crop, model },
                );
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
          <p className="mt-2 text-xs text-neutral-500">
            Keep Unreal closed while it runs; they share the GPU.
            {multiview &&
              " Multi-view takes about 5 minutes and 11 GB of GPU memory. Use views from the same shoot as the front," +
                " each showing the whole head, back of the skull included."}
            {facemodel &&
              " The face model takes about 5.5 minutes (7 with the skin texture): a rough TRELLIS.2 head first, then Google's GNM head fitted to" +
                " the photos, then the skin texture baked from the photos. Side and three-quarter photos shape the" +
                " skull and fill in the texture, and the other three-quarter, other side and back fill in the rest; large photos from one" +
                " evenly lit shoot give the cleanest skin."}
          </p>
        </>
      )}
    </Section>
  );
}

export function Inspector({ actor, selection, onSelect, onChanged, onError, multiviewFound, facemodelFound }: Props) {
  if (!actor) return <aside className="w-80 shrink-0 border-l border-edge bg-panel" />;

  const reference = selection?.kind === "reference" ? actor.references.find((r) => r.id === selection.id) : undefined;
  const generation = reference ? undefined : shownGeneration(actor, selection);
  const job = generation?.job_id ? actor.jobs.find((j) => j.id === generation.job_id) : undefined;

  return (
    <aside className="flex w-80 shrink-0 flex-col overflow-y-auto border-l border-edge bg-panel">
      <GenerateForm
        actor={actor}
        multiviewFound={multiviewFound}
        facemodelFound={facemodelFound}
        onChanged={onChanged}
        onError={onError}
      />

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
          <Field label="Method">
            <span>{MODEL_NAMES[generation.record.model ?? "trellis2"]}</span>
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
