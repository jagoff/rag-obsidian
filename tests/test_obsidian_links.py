"""Tests para `to_obsidian_url` y `convert_obsidian_links`.

Cubren el rewrite de citaciones en --plain → URLs `obsidian://` clickeables
en chat surfaces (WhatsApp). El rewrite tiene que:
  - encodear paths con espacios y caracteres especiales
  - preservar la label en `[Label](path.md)` como prefijo "Label: <url>"
  - manejar la forma bare `[path.md]` (default de command-r) sin doble-procesar
  - dejar texto sin matches intacto
"""
import rag


def test_to_obsidian_url_basic():
    url = rag.to_obsidian_url("02-Areas/Musica/Sal/Letra - Sal.md", vault_name="Notes")
    assert url.startswith("obsidian://open?vault=Notes&file=")
    # Espacios encodean a %20, slashes preservados.
    assert "02-Areas/Musica/Sal/Letra%20-%20Sal.md" in url


def test_to_obsidian_url_default_vault_name():
    # Sin vault_name explícito, usa VAULT_PATH.name.
    url = rag.to_obsidian_url("a/b.md")
    assert f"vault={rag.VAULT_PATH.name}" in url or f"vault={rag.urllib.parse.quote(rag.VAULT_PATH.name, safe='')}" in url


def test_to_obsidian_url_special_chars_in_path():
    url = rag.to_obsidian_url("Explorando (otras)/Tema #1.md", vault_name="V")
    # Paréntesis y # se encodean (URL-safe sin slashes incluye () # en quote default,
    # pero safe='/' no los preserva).
    assert "Explorando%20%28otras%29" in url
    assert "Tema%20%231.md" in url


def test_convert_labeled_link():
    text = "Mirá [Letra de Sal](02-Areas/Musica/Sal/Letra - Sal.md) y listo."
    out = rag.convert_obsidian_links(text, vault_name="Notes")
    assert "[Letra de Sal](" not in out
    assert "Letra de Sal: obsidian://open?vault=Notes&file=02-Areas/Musica/Sal/Letra%20-%20Sal.md" in out
    assert " y listo." in out


def test_convert_bare_path():
    text = "Ver [02-Areas/Musica/Sal.md] para detalles."
    out = rag.convert_obsidian_links(text, vault_name="Notes")
    assert "[02-Areas/Musica/Sal.md]" not in out
    assert "obsidian://open?vault=Notes&file=02-Areas/Musica/Sal.md" in out


def test_convert_mixed_formats():
    text = (
        "Primero [Foo](a/foo.md) después [b/bar.md] "
        "y [Baz Qux](c/baz qux.md)."
    )
    out = rag.convert_obsidian_links(text, vault_name="V")
    assert "Foo: obsidian://open?vault=V&file=a/foo.md" in out
    assert "obsidian://open?vault=V&file=b/bar.md" in out
    assert "Baz Qux: obsidian://open?vault=V&file=c/baz%20qux.md" in out


def test_convert_no_links_passes_through():
    text = "Texto sin links a notas, solo prosa."
    assert rag.convert_obsidian_links(text, vault_name="V") == text


def test_convert_does_not_double_process_labeled_then_bare():
    # `[Label](path.md)` matchea NOTE_LINK_RE; el inner `path.md` NO debe
    # ser tomado por BARE_PATH_RE en el segundo pass (consumed-spans guard).
    text = "[Label](path.md)"
    out = rag.convert_obsidian_links(text, vault_name="V")
    # Solo una URL, no dos.
    assert out.count("obsidian://") == 1
    assert out == "Label: obsidian://open?vault=V&file=path.md"


def test_convert_preserves_paths_with_balanced_parens():
    # NOTE_LINK_RE acepta single-level balanced parens dentro del path
    # (ver línea 52 de rag.py).
    text = "[Otras](02-Areas/Explorando (otras)/X.md) listo"
    out = rag.convert_obsidian_links(text, vault_name="V")
    assert "Otras: obsidian://open?vault=V&file=02-Areas/Explorando%20%28otras%29/X.md" in out
    assert " listo" in out


def test_convert_multiline_text():
    text = "Linea 1 [a.md]\nLinea 2 [Foo](b.md)\nLinea 3 sin link."
    out = rag.convert_obsidian_links(text, vault_name="V")
    assert "obsidian://open?vault=V&file=a.md" in out
    assert "Foo: obsidian://open?vault=V&file=b.md" in out
    assert "Linea 3 sin link." in out
