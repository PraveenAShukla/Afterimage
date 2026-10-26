"""A shell-exported key that differs from .env must stop a run (the wrong-account billing)."""
import pytest

from collusion import env as E


def test_conflicting_shell_key_refused(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("X_API_KEY=project\nY_API_KEY=same\n")
    monkeypatch.setattr(E, "REPO", tmp_path)
    monkeypatch.setenv("X_API_KEY", "shell")
    monkeypatch.setenv("Y_API_KEY", "same")
    monkeypatch.delenv("ALLOW_SHELL_KEYS", raising=False)
    assert E.key_conflicts(f) == ["X_API_KEY"]
    with pytest.raises(E.KeyConflict, match="X_API_KEY"):
        E.load_env()
    monkeypatch.setenv("ALLOW_SHELL_KEYS", "1")
    E.load_env()
    monkeypatch.delenv("X_API_KEY")
    monkeypatch.delenv("ALLOW_SHELL_KEYS")
    E.load_env()
