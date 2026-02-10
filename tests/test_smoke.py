"""Smoke test: verify that vibroglass is importable."""

from __future__ import annotations


def test_import() -> None:
    import vibroglass

    assert vibroglass.__version__ == "0.1.0"


def test_public_api() -> None:
    from vibroglass import (
        LanczosOptions,
        VibrationalSpectra,
        VibrationalSystem,
        load_dynmat_and_atoms,
    )

    assert callable(load_dynmat_and_atoms)
    assert callable(VibrationalSystem)
    assert callable(VibrationalSpectra)
    assert callable(LanczosOptions)
