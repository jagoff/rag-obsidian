from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _declared_packages() -> set[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return set(data["tool"]["setuptools"]["packages"])


def test_split_subpackages_are_in_setuptools_package_list():
    packages = _declared_packages()
    expected = {
        "rag.cli",
        "rag.integrations.whatsapp",
        "rag.plists",
        "rag.runtime",
        "rag.runtime.jobs",
        "rag_negotiations",
    }
    missing = sorted(expected - packages)
    assert not missing, f"pyproject.toml no empaqueta: {missing}"


def test_web_static_is_included_as_package_data():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = data["tool"]["setuptools"]["package-data"]
    assert "static/**/*" in package_data["web"]
