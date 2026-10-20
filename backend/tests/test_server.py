import io
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tamo_server.config import Settings  # noqa: E402
from tamo_server.main import create_app  # noqa: E402

FAKE_ENGINE = Path(__file__).with_name("fake_engine.py")


@pytest.fixture
def library(tmp_path):
    return tmp_path / "library" / "projects"


@pytest.fixture
def client(tmp_path, library, monkeypatch):
    settings = Settings(
        library=library, database=tmp_path / "db" / "studio.sqlite", logs=tmp_path / "db" / "logs",
        engine_python=Path(sys.executable), engine_script=FAKE_ENGINE, espalier=tmp_path,
        app_dist=tmp_path / "no-dist", multiview_python=Path(sys.executable), multiview_script=FAKE_ENGINE,
        pixal3d=tmp_path,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def png(width=512, height=512) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (120, 90, 70)).save(buffer, "PNG")
    return buffer.getvalue()


def wait_for(client, job_id, statuses=("done", "failed", "cancelled"), timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in statuses:
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} still {job['status']}")


def make_actor(client, name="Nia Okafor"):
    project = client.post("/api/projects", json={"name": "Tamo"}).json()
    actor = client.post(f"/api/projects/{project['id']}/actors", json={"name": name}).json()
    reference = client.post(f"/api/actors/{actor['id']}/references",
                            files={"file": ("Nia front.png", png(), "image/png")}, data={"view": "front"}).json()
    return project, actor, reference


def test_generation_end_to_end(client, library):
    project, actor, reference = make_actor(client)
    assert actor["slug"] == "nia_okafor"
    assert Path(reference["path"]).parent == library / "tamo" / "actors" / "nia_okafor" / "references"

    job = client.post(f"/api/actors/{actor['id']}/generations", json={"reference_id": reference["id"]}).json()
    assert job["settings"] == {"texture": True, "pipeline": "1024_cascade", "seed": 42, "smooth": 1.0, "crop": True,
                               "model": "trellis2"}
    job = wait_for(client, job["id"])
    assert job["status"] == "done", job["error"]
    assert job["progress"] == 1

    details = client.get(f"/api/actors/{actor['id']}").json()
    [generation] = details["generations"]
    assert generation["job_id"] == job["id"]
    assert generation["record"]["result"] == "ok"
    assert {"model.glb", "head_metahuman.fbx", "run.json"} <= set(generation["files"])
    fbx = client.get(f"/api/generations/{generation['id']}/files/head_metahuman.fbx")
    assert fbx.status_code == 200 and fbx.content == b"FBX"
    assert client.get(f"/api/generations/{generation['id']}/files/..%2Frun.json").status_code == 404
    assert "Exporting the MetaHuman" in client.get(f"/api/jobs/{job['id']}/log").text

    assert client.post(f"/api/actors/{actor['id']}/preferred",
                       data={"generation_id": generation["id"]}).json()["preferred_generation_id"] == generation["id"]


def test_multiview_uses_the_actors_turned_views(client, library):
    _, actor, front = make_actor(client)
    url = f"/api/actors/{actor['id']}/generations"
    assert client.post(url, json={"reference_id": front["id"], "model": "pixal3d"}).status_code == 422   # no turned views
    assert client.post(url, json={"reference_id": front["id"], "side_id": front["id"]}).status_code == 422
    side = client.post(f"/api/actors/{actor['id']}/references",
                       files={"file": ("Nia side.png", png(), "image/png")}, data={"view": "side"}).json()
    assert client.post(url, json={"reference_id": front["id"], "model": "pixal3d",
                                  "pipeline": "1024"}).status_code == 422

    job = client.post(url, json={"reference_id": front["id"], "model": "pixal3d"}).json()
    assert job["settings"]["model"] == "pixal3d"
    assert job["settings"]["side_id"] == side["id"] and job["settings"]["three_quarter_id"] is None
    job = wait_for(client, job["id"])
    assert job["status"] == "done", job["error"]
    assert "/generations/" in job["run_dir"] and job["run_dir"].split("/")[-1].endswith("-multiview-textured")
    log = client.get(f"/api/jobs/{job['id']}/log").text
    assert f"--front {front['path']}" in log and f"--side {side['path']}" in log and "--three-quarter" not in log
    [generation] = client.get(f"/api/actors/{actor['id']}").json()["generations"]
    assert generation["record"]["model"] == "pixal3d"


def test_turned_view_in_use_cant_be_deleted(client, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE", "slow")
    _, actor, front = make_actor(client)
    side = client.post(f"/api/actors/{actor['id']}/references",
                       files={"file": ("Nia side.png", png(), "image/png")}, data={"view": "side"}).json()
    job = client.post(f"/api/actors/{actor['id']}/generations",
                      json={"reference_id": front["id"], "model": "pixal3d"}).json()
    assert client.delete(f"/api/references/{side['id']}").status_code == 409
    client.post(f"/api/jobs/{job['id']}/cancel")
    wait_for(client, job["id"])
    assert client.delete(f"/api/references/{side['id']}").status_code == 200


def test_multiview_progress_follows_pixal3ds_bars():
    from tamo_server.engine import Pixal3DEngine, Progress
    engine, progress, seen = Pixal3DEngine(None), Progress("Starting", 0.0), []
    lines = ["Removing background with BiRefNet...", "Cropped to head and neck.",
             "GPU memory in use before generation: 1007 MiB"]
    for bar in ("Sampling sparse structure (proj)", "Sampling shape SLat (proj)",
                "Sampling HR shape SLat (proj, 1024)", "Sampling texture SLat (proj)"):
        lines += [f"{bar}: {n}%|##| {n}/100" for n in (0, 50, 100, 100)]
    lines += ["generated {'mesh': 1}", "Sampling attributes", "exported /x/model.glb",
              "Exporting the MetaHuman-aligned head with Blender...", "Result:       ok"]
    for line in lines:
        moved = engine.progress(line, progress)
        if moved:
            progress = moved
            seen.append((progress.stage, progress.fraction))
    fractions = [f for _, f in seen]
    assert fractions == sorted(fractions)
    stages = [s for s, _ in seen]
    for stage in ("Generating: rough shape", "Generating: shape detail", "Generating: texture", "Baking the texture"):
        assert stage in stages
    assert max(f for s, f in seen if s.startswith("Generating")) <= 0.52


def test_failed_job_keeps_the_error(client, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE", "fail")
    _, actor, reference = make_actor(client)
    job = client.post(f"/api/actors/{actor['id']}/generations", json={"reference_id": reference["id"]}).json()
    job = wait_for(client, job["id"])
    assert job["status"] == "failed"
    assert "CUDA out of memory" in job["error"]
    assert client.get("/api/health").json()["ok"]


def test_cancel_running_job(client, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE", "slow")
    _, actor, reference = make_actor(client)
    job = client.post(f"/api/actors/{actor['id']}/generations", json={"reference_id": reference["id"]}).json()
    wait_for(client, job["id"], statuses=("running",))
    assert client.post(f"/api/jobs/{job['id']}/cancel").json()["status"] in ("running", "cancelled")
    assert wait_for(client, job["id"])["status"] == "cancelled"
    assert client.post(f"/api/jobs/{job['id']}/cancel").status_code == 409


def test_progress_moves_through_stages(client, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE", "slow")
    _, actor, reference = make_actor(client)
    job = client.post(f"/api/actors/{actor['id']}/generations", json={"reference_id": reference["id"]}).json()
    seen = set()
    deadline = time.time() + 20
    while time.time() < deadline:
        current = client.get(f"/api/jobs/{job['id']}").json()
        seen.add((current["stage"], round(current["progress"], 2)))
        if current["status"] == "done":
            break
        time.sleep(0.05)
    stages = {stage for stage, _ in seen}
    assert {"Generating: shape", "Generating: shape detail", "Generating: texture"} <= stages
    # The cascade's second shape pass restarts at 0% but lands in its own band, after the first.
    detail = [p for stage, p in seen if stage == "Generating: shape detail"]
    assert detail and min(detail) >= 0.57 and max(detail) <= 0.66


def test_rejects_bad_uploads_and_settings(client):
    _, actor, reference = make_actor(client)
    bad = client.post(f"/api/actors/{actor['id']}/references", files={"file": ("x.png", b"not an image", "image/png")})
    assert bad.status_code == 422
    tiny = client.post(f"/api/actors/{actor['id']}/references", files={"file": ("x.png", png(64, 64), "image/png")})
    assert tiny.status_code == 422
    wrong_view = client.post(f"/api/actors/{actor['id']}/references", files={"file": ("x.png", png(), "image/png")},
                             data={"view": "selfie"})
    assert wrong_view.status_code == 422
    bad_pipeline = client.post(f"/api/actors/{actor['id']}/generations",
                               json={"reference_id": reference["id"], "pipeline": "4096"})
    assert bad_pipeline.status_code == 422
    assert client.post("/api/projects", json={"name": "Tamo"}).status_code == 409


def test_existing_runs_on_disk_are_imported(tmp_path, library):
    run = library / "tamo" / "actors" / "amara_okafor" / "generations" / "20261018-205938-textured"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"result": "ok", "started": "2026-10-18T20:59:38"}))
    (run / "model.glb").write_bytes(b"glTF")
    references = library / "tamo" / "actors" / "amara_okafor" / "references"
    references.mkdir()
    Image.new("RGB", (300, 300)).save(references / "headshot.png")

    settings = Settings(
        library=library, database=tmp_path / "studio.sqlite", logs=tmp_path / "logs",
        engine_python=Path(sys.executable), engine_script=FAKE_ENGINE, espalier=tmp_path, app_dist=tmp_path / "x",
    )
    with TestClient(create_app(settings)) as client:
        [project] = client.get("/api/projects").json()
        [actor] = client.get(f"/api/projects/{project['id']}/actors").json()
        assert actor["name"] == "Amara Okafor" and actor["generation_count"] == 1
        details = client.get(f"/api/actors/{actor['id']}").json()
        assert details["references"][0]["view"] == "unsorted"
        assert client.post("/api/rescan").json() == {"added_generations": 0}


def test_restart_requeues_interrupted_jobs(tmp_path, library, monkeypatch):
    settings = Settings(
        library=library, database=tmp_path / "studio.sqlite", logs=tmp_path / "logs",
        engine_python=Path(sys.executable), engine_script=FAKE_ENGINE, espalier=tmp_path, app_dist=tmp_path / "x",
    )
    monkeypatch.setenv("FAKE_ENGINE", "slow")
    with TestClient(create_app(settings)) as client:
        _, actor, reference = make_actor(client)
        job = client.post(f"/api/actors/{actor['id']}/generations", json={"reference_id": reference["id"]}).json()
        wait_for(client, job["id"], statuses=("running",))
    # The server stopped mid-run; the next start picks the job up again.
    monkeypatch.setenv("FAKE_ENGINE", "ok")
    with TestClient(create_app(settings)) as client:
        done = wait_for(client, job["id"])
        assert done["status"] == "done" and done["attempts"] == 2


def trashed(library):
    trash = library.parent / "trash"
    return sorted(p.relative_to(trash).as_posix().split("/", 1)[1] for p in trash.glob("*/**/*") if p.is_file())


def test_delete_version_image_and_actor_moves_them_to_the_trash(client, library):
    project, actor, reference = make_actor(client)
    job = client.post(f"/api/actors/{actor['id']}/generations", json={"reference_id": reference["id"]}).json()
    wait_for(client, job["id"])
    [generation] = client.get(f"/api/actors/{actor['id']}").json()["generations"]
    client.post(f"/api/actors/{actor['id']}/preferred", data={"generation_id": generation["id"]})
    run_name = Path(generation["run_dir"]).name

    assert client.delete(f"/api/generations/{generation['id']}").status_code == 200
    details = client.get(f"/api/actors/{actor['id']}").json()
    assert details["generations"] == [] and details["preferred_generation_id"] is None
    assert not Path(generation["run_dir"]).exists()
    assert f"tamo/actors/nia_okafor/generations/{run_name}/model.glb" in trashed(library)
    assert client.post("/api/rescan").json() == {"added_generations": 0}   # the trash isn't scanned back in

    # The image's jobs go with it; the image itself goes to the trash.
    assert client.delete(f"/api/references/{reference['id']}").status_code == 200
    details = client.get(f"/api/actors/{actor['id']}").json()
    assert details["references"] == [] and details["jobs"] == []
    assert any(p.endswith("Nia_front.png") for p in trashed(library))

    assert client.delete(f"/api/actors/{actor['id']}").status_code == 200
    assert client.get(f"/api/actors/{actor['id']}").status_code == 404
    assert not (library / "tamo" / "actors" / "nia_okafor").exists()
    # The same name can be used again afterwards.
    assert client.post(f"/api/projects/{project['id']}/actors", json={"name": "Nia Okafor"}).status_code == 201
    assert client.delete(f"/api/projects/{project['id']}").status_code == 200
    assert client.get("/api/projects").json() == []
    assert client.delete(f"/api/projects/{project['id']}").status_code == 404


def test_delete_refuses_while_a_job_runs_and_cancel_trashes_the_partial_run(client, library, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE", "slow")
    project, actor, reference = make_actor(client)
    job = client.post(f"/api/actors/{actor['id']}/generations", json={"reference_id": reference["id"]}).json()
    job = wait_for(client, job["id"], statuses=("running",))
    deadline = time.time() + 10
    while not Path(job["run_dir"]).exists() and time.time() < deadline:
        time.sleep(0.05)
    assert Path(job["run_dir"]).exists()
    for path in (f"/api/references/{reference['id']}", f"/api/actors/{actor['id']}", f"/api/projects/{project['id']}"):
        response = client.delete(path)
        assert response.status_code == 409 and "Cancel it first" in response.json()["detail"]
    client.post(f"/api/jobs/{job['id']}/cancel")
    wait_for(client, job["id"])
    assert not Path(job["run_dir"]).exists()
    [partial] = (library.parent / "trash").glob("*/tamo/actors/nia_okafor/generations/*")
    assert partial.name == Path(job["run_dir"]).name
    assert client.delete(f"/api/actors/{actor['id']}").status_code == 200


def test_update_pulls_then_restarts_and_waits_for_running_jobs(client, monkeypatch):
    from tamo_server import updates

    calls = []
    monkeypatch.setattr(updates, "pull", lambda: calls.append("pull") or {"from": "aaa", "commit": "bbb"})
    monkeypatch.setattr(updates, "restart_soon", lambda stop: calls.append("restart"))
    assert {"commit", "shell_version"} <= set(client.get("/api/health").json())

    monkeypatch.setenv("FAKE_ENGINE", "slow")
    _, actor, reference = make_actor(client)
    job = client.post(f"/api/actors/{actor['id']}/generations", json={"reference_id": reference["id"]}).json()
    assert client.post("/api/update").status_code == 409 and calls == []
    client.post(f"/api/jobs/{job['id']}/cancel")
    wait_for(client, job["id"])

    assert client.post("/api/update").json()["commit"] == "bbb"
    assert calls == ["pull", "restart"]

    def offline():
        raise updates.UpdateError("Could not resolve host: github.com")
    monkeypatch.setattr(updates, "pull", offline)
    response = client.post("/api/update")
    assert response.status_code == 502 and "github.com" in response.json()["detail"]
