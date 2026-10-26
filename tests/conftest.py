"""Tests run mock games only; a key the developer's shell exports must not trip the .env key-conflict guard here
(tests/test_env_keys.py checks the guard itself with ALLOW_SHELL_KEYS removed)."""
import pytest


@pytest.fixture(autouse=True)
def _allow_shell_keys(monkeypatch):
    monkeypatch.setenv("ALLOW_SHELL_KEYS", "1")
