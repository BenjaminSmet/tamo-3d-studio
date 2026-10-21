"""Tamo 3D Studio's local server: the interface talks to this, and this talks to the engine.

Run it inside WSL (it only listens on this PC):

    uvicorn --factory tamo_server.main:create_app --host 127.0.0.1 --port 8765
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import Settings
from .db import Database, generation_view, job_view, now
from .engine import TURNED_VIEWS, GnmEngine, Pixal3DEngine, Trellis2Engine
from . import updates
from .jobs import JobQueue
from .library import (VIEWS, InvalidImage, actor_dir, ensure_actor, ensure_project, move_to_trash,
                      save_reference, scan, slugify, trash_dir, windows_path)

# Files of a run the interface may fetch, and nothing else from the disk.
RUN_FILES = {
    "model.glb", "model.obj", "head_metahuman.fbx", "input_rgba.png", "input_cutout.png",
    "run.json", "espalier.log", "vram.csv",
}

ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",   # interface dev server
    "tauri://localhost", "http://tauri.localhost", "https://tauri.localhost",   # packaged app
]


class NewProject(BaseModel):
    name: str


class NewActor(BaseModel):
    name: str


class NewGeneration(BaseModel):
    reference_id: int
    model: str = "trellis2"            # "pixal3d": multi-view, with a side and/or three-quarter image; "gnm": face model
    side_id: int | None = None         # multi-view and face model; the actor's newest of each view when left out
    three_quarter_id: int | None = None
    texture: bool | None = None
    pipeline: str | None = None
    seed: int | None = None
    smooth: float | None = None
    crop: bool | None = None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    db = Database(settings.database)
    engines = {e.name: e for e in (Trellis2Engine(settings), Pixal3DEngine(settings), GnmEngine(settings))}
    queue = JobQueue(db, engines, settings.library, settings.logs)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        scan(db, settings.library)
        queue.start()
        yield
        queue.stop()

    app = FastAPI(title="Tamo 3D Studio", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])
    app.state.db, app.state.queue = db, queue

    def project_or_404(project_id: int) -> dict:
        project = db.one("SELECT * FROM projects WHERE id = ?", (project_id,))
        if not project:
            raise HTTPException(404, "No such project.")
        return project

    def actor_or_404(actor_id: int) -> dict:
        actor = db.one("SELECT a.*, p.slug AS project_slug FROM actors a JOIN projects p ON p.id = a.project_id "
                       "WHERE a.id = ?", (actor_id,))
        if not actor:
            raise HTTPException(404, "No such actor.")
        return actor

    def refuse_if_busy(where: str, params: tuple, what: str) -> None:
        if db.one(f"SELECT j.id FROM jobs j JOIN actors a ON a.id = j.actor_id "
                  f"WHERE j.status IN ('queued', 'running') AND {where}", params):
            raise HTTPException(409, f"{what} has a generation queued or running. Cancel it first.")

    def generation_out(generation: dict) -> dict:
        view = generation_view(generation)
        run_dir = Path(view["run_dir"])
        view["files"] = sorted(f for f in RUN_FILES if (run_dir / f).is_file())
        view["windows_path"] = windows_path(str(run_dir))
        return view

    def reference_out(reference: dict) -> dict:
        return {**reference, "windows_path": windows_path(reference["path"])}

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "engine": "trellis2",
            "engine_found": engines["trellis2"].found(),
            "multiview_found": engines["pixal3d"].found(),
            "facemodel_found": engines["gnm"].found(),
            "library": str(settings.library),
            "library_windows_path": windows_path(str(settings.library)),
            "trash_windows_path": windows_path(str(trash_dir(settings.library))),
            **updates.current(),
        }

    @app.post("/api/update/check")
    def check_for_update():
        try:
            return updates.check()
        except updates.UpdateError as error:
            raise HTTPException(502, f"Couldn't check GitHub for updates: {error}")

    @app.post("/api/update")
    def update():
        if db.one("SELECT id FROM jobs WHERE status IN ('queued', 'running')"):
            raise HTTPException(409, "A generation is queued or running. Let it finish, or cancel it, then update.")
        try:
            result = updates.pull()
        except updates.UpdateError as error:
            raise HTTPException(502, f"Couldn't update: {error}")
        updates.restart_soon(queue.stop)
        return result

    @app.post("/api/rescan")
    def rescan():
        return {"added_generations": scan(db, settings.library)}

    @app.get("/api/projects")
    def projects():
        return db.all("SELECT p.*, (SELECT COUNT(*) FROM actors a WHERE a.project_id = p.id) AS actor_count "
                      "FROM projects p ORDER BY p.name")

    @app.post("/api/projects", status_code=201)
    def create_project(body: NewProject):
        try:
            slug = slugify(body.name)
        except ValueError as error:
            raise HTTPException(422, str(error))
        if db.one("SELECT id FROM projects WHERE slug = ?", (slug,)):
            raise HTTPException(409, "A project with that name already exists.")
        project_id = ensure_project(db, slug, body.name.strip())
        (settings.library / slug / "actors").mkdir(parents=True, exist_ok=True)
        return project_or_404(project_id)

    @app.get("/api/projects/{project_id}/actors")
    def actors(project_id: int):
        project_or_404(project_id)
        return db.all(
            "SELECT a.*, "
            "(SELECT r.id FROM reference_images r WHERE r.actor_id = a.id AND r.view = 'front' "
            " ORDER BY r.id DESC LIMIT 1) AS front_reference_id, "
            "(SELECT COUNT(*) FROM generations g WHERE g.actor_id = a.id) AS generation_count "
            "FROM actors a WHERE a.project_id = ? ORDER BY a.name", (project_id,))

    @app.post("/api/projects/{project_id}/actors", status_code=201)
    def create_actor(project_id: int, body: NewActor):
        project = project_or_404(project_id)
        try:
            slug = slugify(body.name)
        except ValueError as error:
            raise HTTPException(422, str(error))
        if db.one("SELECT id FROM actors WHERE project_id = ? AND slug = ?", (project_id, slug)):
            raise HTTPException(409, "An actor with that name already exists in this project.")
        actor_id = ensure_actor(db, project_id, slug, body.name.strip())
        actor_dir(settings.library, project["slug"], slug).mkdir(parents=True, exist_ok=True)
        return actor_or_404(actor_id)

    @app.get("/api/actors/{actor_id}")
    def actor(actor_id: int):
        found = actor_or_404(actor_id)
        found["references"] = [reference_out(r) for r in db.all(
            "SELECT * FROM reference_images WHERE actor_id = ? ORDER BY id DESC", (actor_id,))]
        found["generations"] = [generation_out(g) for g in db.all(
            "SELECT * FROM generations WHERE actor_id = ? ORDER BY created DESC, id DESC", (actor_id,))]
        found["jobs"] = [job_view(j) for j in db.all(
            "SELECT * FROM jobs WHERE actor_id = ? ORDER BY id DESC LIMIT 20", (actor_id,))]
        return found

    @app.post("/api/actors/{actor_id}/references", status_code=201)
    async def upload_reference(actor_id: int, file: UploadFile = File(...), view: str = Form("front")):
        found = actor_or_404(actor_id)
        if view not in VIEWS:
            raise HTTPException(422, f"view must be one of {', '.join(VIEWS)}")
        data = await file.read()
        try:
            path, width, height = save_reference(settings.library, found["project_slug"], found["slug"], view,
                                                 file.filename or "image", data)
        except InvalidImage as error:
            raise HTTPException(422, str(error))
        reference_id = db.insert("reference_images", actor_id=actor_id, view=view, path=str(path),
                                 width=width, height=height, created=now())
        return reference_out(db.one("SELECT * FROM reference_images WHERE id = ?", (reference_id,)))

    @app.patch("/api/references/{reference_id}")
    def set_reference_view(reference_id: int, view: str = Form(...)):
        if view not in VIEWS:
            raise HTTPException(422, f"view must be one of {', '.join(VIEWS)}")
        if not db.one("SELECT id FROM reference_images WHERE id = ?", (reference_id,)):
            raise HTTPException(404, "No such reference image.")
        db.update("reference_images", reference_id, view=view)
        return reference_out(db.one("SELECT * FROM reference_images WHERE id = ?", (reference_id,)))

    @app.get("/api/references/{reference_id}/image")
    def reference_image(reference_id: int):
        reference = db.one("SELECT path FROM reference_images WHERE id = ?", (reference_id,))
        if not reference or not Path(reference["path"]).is_file():
            raise HTTPException(404, "No such reference image.")
        return FileResponse(reference["path"])

    @app.post("/api/actors/{actor_id}/generations", status_code=202)
    def start_generation(actor_id: int, body: NewGeneration):
        actor_or_404(actor_id)
        reference = db.one("SELECT * FROM reference_images WHERE id = ? AND actor_id = ?",
                           (body.reference_id, actor_id))
        if not reference:
            raise HTTPException(404, "That reference image doesn't belong to this actor.")
        engine = engines.get(body.model)
        if engine is None:
            raise HTTPException(422, f"model must be one of {', '.join(engines)}")
        options = body.model_dump(exclude={"reference_id", "model"})
        if engine.views:
            for view in engine.views:
                key = f"{view}_id"
                if options[key] is None:
                    newest = db.one("SELECT id FROM reference_images WHERE actor_id = ? AND view = ? "
                                    "ORDER BY id DESC LIMIT 1", (actor_id, view))
                    options[key] = newest["id"] if newest else None
                elif not db.one("SELECT id FROM reference_images WHERE id = ? AND actor_id = ?",
                                (options[key], actor_id)):
                    raise HTTPException(404, f"That {view.replace('_', '-')} image doesn't belong to this actor.")
            if engine.needs_turned_view and all(options[f"{view}_id"] is None for view in engine.views):
                raise HTTPException(422, "Multi-view needs a side or three-quarter image of this actor.")
        else:
            for view in TURNED_VIEWS:
                if options.pop(f"{view}_id") is not None:
                    raise HTTPException(422, "Only multi-view and the face model use side and three-quarter images.")
        try:
            options = {**engine.validate(options), "model": engine.name}
        except ValueError as error:
            raise HTTPException(422, str(error))
        job_id = queue.submit(actor_id, reference["id"], options)
        return job_view(db.one("SELECT * FROM jobs WHERE id = ?", (job_id,)))

    @app.get("/api/jobs")
    def jobs(active: bool = False):
        where = "WHERE j.status IN ('queued', 'running')" if active else ""
        rows = db.all(f"SELECT j.*, a.name AS actor_name FROM jobs j JOIN actors a ON a.id = j.actor_id {where} "
                      "ORDER BY j.id DESC LIMIT 50")
        return [job_view(j) for j in rows]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: int):
        found = db.one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        if not found:
            raise HTTPException(404, "No such job.")
        return job_view(found)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: int):
        if not db.one("SELECT id FROM jobs WHERE id = ?", (job_id,)):
            raise HTTPException(404, "No such job.")
        if not queue.cancel(job_id):
            raise HTTPException(409, "That job has already finished.")
        return job_view(db.one("SELECT * FROM jobs WHERE id = ?", (job_id,)))

    @app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
    def job_log(job_id: int, lines: int = 200):
        path = queue.log_path(job_id)
        if not path.is_file():
            return ""
        return "\n".join(path.read_text(errors="replace").splitlines()[-max(1, min(lines, 5000)):])

    @app.get("/api/generations/{generation_id}")
    def generation(generation_id: int):
        found = db.one("SELECT * FROM generations WHERE id = ?", (generation_id,))
        if not found:
            raise HTTPException(404, "No such generation.")
        return generation_out(found)

    @app.get("/api/generations/{generation_id}/files/{name}")
    def generation_file(generation_id: int, name: str):
        found = db.one("SELECT run_dir FROM generations WHERE id = ?", (generation_id,))
        if not found or name not in RUN_FILES:
            raise HTTPException(404, "No such file.")
        path = Path(found["run_dir"]) / name
        if not path.is_file():
            raise HTTPException(404, "That file wasn't produced by this run.")
        return FileResponse(path, filename=name if name.endswith((".fbx", ".obj", ".glb")) else None)

    @app.post("/api/actors/{actor_id}/preferred")
    def set_preferred(actor_id: int, generation_id: int = Form(...)):
        actor_or_404(actor_id)
        if not db.one("SELECT id FROM generations WHERE id = ? AND actor_id = ?", (generation_id, actor_id)):
            raise HTTPException(404, "That generation doesn't belong to this actor.")
        db.update("actors", actor_id, preferred_generation_id=generation_id)
        return actor_or_404(actor_id)

    # Deleting moves files to library/trash and forgets them here; nothing is erased from the disk.
    # Jobs that used a deleted image go too (each version keeps its own settings and input in run.json).

    @app.delete("/api/generations/{generation_id}")
    def delete_generation(generation_id: int):
        found = db.one("SELECT * FROM generations WHERE id = ?", (generation_id,))
        if not found:
            raise HTTPException(404, "No such generation.")
        db.transaction([
            ("UPDATE actors SET preferred_generation_id = NULL WHERE preferred_generation_id = ?", (generation_id,)),
            ("DELETE FROM generations WHERE id = ?", (generation_id,)),
        ])
        move_to_trash(settings.library, Path(found["run_dir"]))
        return {"deleted": generation_id}

    @app.delete("/api/references/{reference_id}")
    def delete_reference(reference_id: int):
        found = db.one("SELECT * FROM reference_images WHERE id = ?", (reference_id,))
        if not found:
            raise HTTPException(404, "No such reference image.")
        refuse_if_busy("(j.reference_id = ? OR json_extract(j.settings, '$.side_id') = ? "
                       "OR json_extract(j.settings, '$.three_quarter_id') = ?)", (reference_id,) * 3, "This image")
        db.transaction([
            ("UPDATE generations SET job_id = NULL WHERE job_id IN (SELECT id FROM jobs WHERE reference_id = ?)",
             (reference_id,)),
            ("DELETE FROM jobs WHERE reference_id = ?", (reference_id,)),
            ("DELETE FROM reference_images WHERE id = ?", (reference_id,)),
        ])
        move_to_trash(settings.library, Path(found["path"]))
        return {"deleted": reference_id}

    def forget_actors(where: str, params: tuple) -> list[tuple[str, tuple]]:
        actors_sql = f"SELECT id FROM actors WHERE {where}"
        return [
            (f"DELETE FROM generations WHERE actor_id IN ({actors_sql})", params),
            (f"DELETE FROM jobs WHERE actor_id IN ({actors_sql})", params),
            (f"DELETE FROM reference_images WHERE actor_id IN ({actors_sql})", params),
            (f"DELETE FROM actors WHERE {where}", params),
        ]

    @app.delete("/api/actors/{actor_id}")
    def delete_actor(actor_id: int):
        found = actor_or_404(actor_id)
        refuse_if_busy("a.id = ?", (actor_id,), "This actor")
        db.transaction(forget_actors("id = ?", (actor_id,)))
        move_to_trash(settings.library, actor_dir(settings.library, found["project_slug"], found["slug"]))
        return {"deleted": actor_id}

    @app.delete("/api/projects/{project_id}")
    def delete_project(project_id: int):
        found = project_or_404(project_id)
        refuse_if_busy("a.project_id = ?", (project_id,), "An actor in this project")
        db.transaction([*forget_actors("project_id = ?", (project_id,)),
                        ("DELETE FROM projects WHERE id = ?", (project_id,))])
        move_to_trash(settings.library, settings.library / found["slug"])
        return {"deleted": project_id}

    if settings.app_dist.is_dir():
        app.mount("/", StaticFiles(directory=settings.app_dist, html=True), name="app")

    return app

