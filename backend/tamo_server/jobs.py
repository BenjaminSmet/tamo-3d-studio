"""Background job queue: one generation at a time, since they share the GPU.

Jobs survive restarts: anything that was running when the server stopped goes back
to the queue and starts again in a fresh run folder. Cancelling stops the whole
process group, Espalier-2 included, and moves the half-finished folder to the trash. A failed job records the end of its log and never
takes the server down.
"""

import json
import os
import signal
import subprocess
import threading
from collections import deque
from pathlib import Path

from .db import Database, now
from .engine import Progress
from .library import move_to_trash, new_run_dir

LOG_TAIL_LINES = 40


class JobQueue:
    def __init__(self, db: Database, engines: dict, library: Path, logs: Path):
        self.db = db
        self.engines = engines   # by name; a job's settings name its engine (TRELLIS.2 when they don't)
        self.library = library
        self.logs = logs
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._running_id: int | None = None
        self._cancelled: set[int] = set()
        self._thread = threading.Thread(target=self._loop, name="job-queue", daemon=True)

    def start(self) -> None:
        self.logs.mkdir(parents=True, exist_ok=True)
        self.db.execute("UPDATE jobs SET status = 'queued', stage = 'Waiting (restarted)', progress = 0 "
                        "WHERE status = 'running'")
        self._thread.start()
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        with self._lock:
            if self._process and self._process.poll() is None:
                self._kill(self._process)
        self._thread.join(timeout=10)

    def submit(self, actor_id: int, reference_id: int, settings: dict) -> int:
        job_id = self.db.insert("jobs", actor_id=actor_id, reference_id=reference_id, settings=json.dumps(settings),
                                status="queued", stage="Waiting", progress=0, created=now())
        self.log_path(job_id).unlink(missing_ok=True)   # ids of deleted jobs get reused
        self._wake.set()
        return job_id

    def cancel(self, job_id: int) -> bool:
        """Cancels a queued or running job. Returns False if it had already finished."""
        with self._lock:
            job = self.db.one("SELECT status FROM jobs WHERE id = ?", (job_id,))
            if not job or job["status"] not in ("queued", "running"):
                return False
            if job["status"] == "queued":
                self.db.update("jobs", job_id, status="cancelled", stage="Cancelled", finished=now())
                return True
            self._cancelled.add(job_id)
            if self._running_id == job_id and self._process and self._process.poll() is None:
                self._kill(self._process)
            return True

    def log_path(self, job_id: int) -> Path:
        return self.logs / f"job-{job_id}.log"

    @staticmethod
    def _kill(process: subprocess.Popen) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            job = self.db.one("SELECT * FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1")
            if job is None:
                self._wake.wait()
                self._wake.clear()
                continue
            try:
                self._run(job)
            except Exception as error:  # a bug here must not stop the queue
                self.db.update("jobs", job["id"], status="failed", stage="Failed",
                               error=f"Internal error: {error}", finished=now())

    def _run(self, job: dict) -> None:
        job_id = job["id"]
        settings = json.loads(job["settings"])
        info = self.db.one(
            "SELECT a.slug AS actor, p.slug AS project, r.path AS image FROM actors a "
            "JOIN projects p ON p.id = a.project_id JOIN reference_images r ON r.id = ? WHERE a.id = ?",
            (job["reference_id"], job["actor_id"]))
        engine = self.engines[settings.get("model", "trellis2")]
        images = {"front": Path(info["image"])} if info else {}
        for view in engine.views:
            reference_id = settings.get(f"{view}_id")
            if reference_id is not None:
                row = self.db.one("SELECT path FROM reference_images WHERE id = ?", (reference_id,))
                images[view] = Path(row["path"]) if row else Path("/missing")
        if info is None or not all(path.is_file() for path in images.values()):
            self.db.update("jobs", job_id, status="failed", stage="Failed",
                           error="A reference image is missing from the library.", finished=now())
            return

        mode = "textured" if settings["texture"] else "geometry"
        run_dir = new_run_dir(self.library, info["project"], info["actor"], engine.run_name + mode)
        cmd = engine.command(images, info["project"], info["actor"], run_dir, settings)
        progress = Progress("Starting", 0.0)
        self.db.update("jobs", job_id, status="running", stage=progress.stage, progress=0, run_dir=str(run_dir),
                       attempts=job["attempts"] + 1, started=now(), error=None)

        tail: deque[str] = deque(maxlen=LOG_TAIL_LINES)
        with self._lock:
            if job_id in self._cancelled:
                self._finish_cancelled(job_id)
                return
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                       errors="replace", start_new_session=True)
            self._process, self._running_id = process, job_id

        with self.log_path(job_id).open("a") as log:
            log.write(f"$ {' '.join(cmd)}\n")
            # tqdm redraws with \r; treat each redraw as a line so progress keeps moving.
            for chunk in process.stdout:
                for line in chunk.replace("\r", "\n").splitlines():
                    if not line.strip():
                        continue
                    log.write(line + "\n")
                    tail.append(line)
                    moved = engine.progress(line, progress)
                    if moved:
                        progress = moved
                        self.db.update("jobs", job_id, stage=progress.stage, progress=round(progress.fraction, 3))
                log.flush()
        exit_code = process.wait()

        with self._lock:
            self._process, self._running_id = None, None
            cancelled = job_id in self._cancelled
        if cancelled:
            self._finish_cancelled(job_id)
            return
        if self._stop.is_set():
            # The server is shutting down mid-run; start this job again next time, in a fresh folder.
            move_to_trash(self.library, run_dir)
            self.db.update("jobs", job_id, status="queued", stage="Waiting (restarted)", progress=0)
            return

        record = self._read_record(run_dir)
        if record is not None:
            self.db.insert("generations", actor_id=job["actor_id"], job_id=job_id, run_dir=str(run_dir),
                           record=json.dumps(record), created=record.get("started") or now())
        if exit_code == 0 and record and record.get("result") == "ok":
            self.db.update("jobs", job_id, status="done", stage="Done", progress=1, finished=now())
        else:
            self.db.update("jobs", job_id, status="failed", stage="Failed", finished=now(),
                           error="\n".join(tail) or f"The engine stopped with exit code {exit_code}.")

    def _finish_cancelled(self, job_id: int) -> None:
        self._cancelled.discard(job_id)
        job = self.db.one("SELECT run_dir FROM jobs WHERE id = ?", (job_id,))
        if job and job["run_dir"]:
            move_to_trash(self.library, Path(job["run_dir"]))   # half-finished; not a version
        self.db.update("jobs", job_id, status="cancelled", stage="Cancelled", finished=now())

    @staticmethod
    def _read_record(run_dir: Path) -> dict | None:
        try:
            return json.loads((run_dir / "run.json").read_text())
        except (OSError, json.JSONDecodeError):
            return None
