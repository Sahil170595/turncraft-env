import json

from turncraft import demo, evaluate, viewer
from turncraft.task_registry import get_task


def test_cli_save_replay_reset_and_evaluation(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(demo, "RUNS_DIR", tmp_path)
    assert demo.main(["--list-tasks"]) == 0
    assert demo.main(["--task", "CANCEL-READY", "--mode", "scripted-good", "--save"]) == 0
    artifact = tmp_path / "scripted-good-CANCEL-READY.json"
    saved = json.loads(artifact.read_text())
    assert saved["initial_db"]["orders"]["ORD-SYN-81"]["status"] == "pending"
    assert saved["final_db"]["orders"]["ORD-SYN-81"]["status"] == "cancelled"
    assert demo.main(["--task", "CANCEL-READY", "--mode", "replay", "--run", str(artifact)]) == 0
    assert (
        demo.main(["--task", "WAIT-FOR-STOCK", "--mode", "scripted-good", "--control", "oracle_refund"]) == 0
    )
    out = tmp_path / "evaluation.json"
    assert evaluate.main(["--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert not report["failed_tasks"]
    assert len(report["results"]) == 34
    assert report["mode"] == "scripted-offline" and report["seed"] is None
    assert get_task("CANCEL-READY").initial_db.orders["ORD-SYN-81"].status == "pending"
    assert "HIDDEN FROM USER" in capsys.readouterr().out


def test_viewer_renders_all_controls_without_opening_browser(tmp_path, monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Browser must not open")

    monkeypatch.setattr(viewer.webbrowser, "open", blocked)
    out = tmp_path / "report.html"
    assert viewer.main(["--all", "--out", str(out)]) == 0
    content = out.read_text(encoding="utf-8")
    assert "Turncraft" in content and "CANCEL-READY" in content
    assert "&lt;" in viewer._esc("<unsafe>")
