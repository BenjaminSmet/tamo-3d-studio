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
        app_dist=tmp_path / "no-dist",
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
    assert job["settings"] == {"texture": True, "pipeline": "1024_cascade", "seed": 42, "smooth": 1.0, "crop": True}
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
    assert "Generating the 3D head" in stages
    assert any(0.1 < p < 0.9 for stage, p in seen if stage == "Generating the 3D head")


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
