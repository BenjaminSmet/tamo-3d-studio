import { useEffect, useState, type ButtonHTMLAttributes, type ReactNode } from "react";

export function Button({
  variant = "plain",
  className = "",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "plain" | "primary" | "danger" }) {
  const styles = {
    plain: "bg-panel-raised hover:bg-edge text-neutral-200",
    primary: "bg-accent hover:bg-blue-500 text-white",
    danger: "bg-red-900/60 hover:bg-red-800 text-red-100",
  }[variant];
  return (
    <button
      className={`rounded px-3 py-1.5 font-medium transition disabled:cursor-not-allowed disabled:opacity-40 ${styles} ${className}`}
      {...props}
    />
  );
}

// A delete link that asks once before acting: the first click turns it into "Delete? Yes / No".
export function ConfirmDelete({ label, onConfirm }: { label: string; onConfirm: () => Promise<unknown> }) {
  const [asking, setAsking] = useState(false);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (!asking) return;
    const timer = setTimeout(() => setAsking(false), 5000);
    return () => clearTimeout(timer);
  }, [asking]);
  if (!asking)
    return (
      <button className="text-xs text-neutral-400 hover:text-red-300" onClick={() => setAsking(true)}>
        {label}
      </button>
    );
  return (
    <span className="flex items-center gap-2 text-xs">
      <span className="text-red-300">{label}?</span>
      <button
        disabled={busy}
        className="rounded bg-red-900/70 px-2 py-0.5 text-red-100 hover:bg-red-800 disabled:opacity-40"
        onClick={async () => {
          setBusy(true);
          try {
            await onConfirm();
          } finally {
            setBusy(false);
            setAsking(false);
          }
        }}
      >
        Yes
      </button>
      <button className="text-neutral-400 hover:text-white" onClick={() => setAsking(false)}>
        No
      </button>
    </span>
  );
}

export function Section({ title, action, children }: { title: string; action?: ReactNode; children: ReactNode }) {
  return (
    <section className="border-b border-edge px-3 py-3">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-[11px] font-semibold uppercase tracking-wider text-neutral-400">{title}</h2>
        {action}
      </div>
      {children}
    </section>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex items-center justify-between gap-3 py-1">
      <span className="text-neutral-400">{label}</span>
      {children}
    </label>
  );
}

export function ProgressBar({ value }: { value: number }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded bg-edge">
      <div className="h-full bg-accent transition-all" style={{ width: `${Math.round(value * 100)}%` }} />
    </div>
  );
}

export function formatSeconds(seconds?: number) {
  if (seconds == null) return "–";
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m ? `${m} min ${s} s` : `${s} s`;
}

export function formatDate(iso?: string | null) {
  if (!iso) return "–";
  const date = new Date(iso);
  return date.toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}
