"""Sanity checks del Sub-chunk 1.1 del package-split CLI: el grupo
``rag vault`` se extrajo de ``rag/__init__.py`` a ``rag/cli/vault.py``.

Estos tests **no** verifican lógica de negocio (eso lo cubre
``tests/test_vaults.py``, que sigue corriendo igual post-extract). Lo
único que validan es la *mecánica del split*:

1. El submódulo importa sin romper (no circulares, no AttributeError).
2. El grupo ``vault`` está registrado contra el ``cli`` Group raíz.
3. Los 5 subcomandos (``add``, ``list``, ``use``, ``current``,
   ``remove``) están todos en el grupo extraído.
4. ``rag vault --help`` corre end-to-end vía CliRunner sin crashear.
5. ``rag.vault`` (la attribute legacy) y ``rag.cli.vault.vault`` (la
   nueva ubicación) apuntan al **mismo** objeto Click — ningún test
   queda accidentalmente apuntando a una copia stale.

Ver ``plans/package-split-2026-04-29.md`` para el contexto del split.
"""
from __future__ import annotations

import click
from click.testing import CliRunner


def test_cli_vault_module_importable():
    """El submódulo extraído carga sin AttributeError / circular."""
    from rag.cli import vault as vault_module
    assert vault_module is not None
    # El módulo expone el grupo y los 5 subcomandos públicos.
    for name in ("vault", "vault_add", "vault_list", "vault_use",
                 "vault_current", "vault_remove"):
        assert hasattr(vault_module, name), name


def test_cli_root_has_vault_group_registered():
    """El grupo ``vault`` está bound al ``cli`` Group raíz post-extract."""
    from rag import cli
    assert isinstance(cli, click.Group), type(cli)
    assert "vault" in cli.commands, sorted(cli.commands.keys())
    # La entry registrada es realmente un Click Group (no un Command).
    assert isinstance(cli.commands["vault"], click.Group)


def test_vault_group_has_five_subcommands():
    """Los 5 subcomandos quedaron todos en el grupo extraído."""
    from rag.cli.vault import vault as vault_grp
    assert isinstance(vault_grp, click.Group)
    expected = {"add", "list", "use", "current", "remove"}
    assert set(vault_grp.commands.keys()) == expected, sorted(vault_grp.commands.keys())
    assert len(vault_grp.commands) == 5


def test_rag_vault_legacy_alias_is_same_object():
    """``rag.vault`` y ``rag.cli.vault.vault`` apuntan al mismo Group.

    Este check protege contra el bug clásico del split: que
    ``rag/__init__.py`` defina (o re-importe) un *segundo* grupo con el
    mismo nombre, dejando los tests legacy apuntando a una copia stale
    sin subcomandos. Probado en producción cuando el commit del
    Sub-chunk 1.1 sale a master.
    """
    import rag
    from rag.cli.vault import vault as vault_grp
    assert rag.vault is vault_grp
    # Y a su vez, lo registrado contra el cli root también es el mismo.
    assert rag.cli.commands["vault"] is vault_grp


def test_rag_vault_help_smoke():
    """``rag vault --help`` corre sin error y lista los 5 subcomandos."""
    from rag import cli
    result = CliRunner().invoke(cli, ["vault", "--help"])
    assert result.exit_code == 0, result.output
    out = result.output
    for sub in ("add", "list", "use", "current", "remove"):
        assert sub in out, (sub, out)
