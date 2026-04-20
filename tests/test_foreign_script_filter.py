"""Post-stream filter enforcing REGLA 0 at the byte level.

The system prompt forbids non-Spanish output but qwen2.5:7b and command-r
leak CJK / Cyrillic / Arabic tokens under context pressure. The web chat
path runs `_strip_foreign_scripts` on every streamed delta so these chars
never reach the user, regardless of what the LLM emits.

If a future refactor removes the filter, these tests fail — that's the
whole point.
"""
from web.server import _strip_foreign_scripts


def test_drops_cjk_ideographs():
    out = _strip_foreign_scripts("consultar otras notas o的目的地信息。")
    assert out == "consultar otras notas o"


def test_drops_japanese_kana():
    assert _strip_foreign_scripts("日本語 test ひらがな カタカナ") == " test  "


def test_drops_cyrillic():
    assert _strip_foreign_scripts("hola русский amigo") == "hola  amigo"


def test_drops_arabic_and_hebrew():
    assert _strip_foreign_scripts("العربية עברית ok") == "  ok"


def test_drops_hangul():
    assert _strip_foreign_scripts("한국어 prueba") == " prueba"


def test_drops_fullwidth_cjk_punctuation():
    assert _strip_foreign_scripts("fin。") == "fin"
    assert _strip_foreign_scripts("a、b") == "ab"


def test_preserves_spanish_accents():
    s = "mañana es niño, Grecia va al lago. ¿está bien?"
    assert _strip_foreign_scripts(s) == s


def test_preserves_emoji():
    s = "coheté 🚀 allowed ✅ 🎉"
    assert _strip_foreign_scripts(s) == s


def test_preserves_ascii_punctuation_and_markdown():
    s = "`código` [enlace](http://x) *italic* **bold** — endash"
    assert _strip_foreign_scripts(s) == s


def test_empty_string_passthrough():
    assert _strip_foreign_scripts("") == ""


def test_only_foreign_chars_becomes_empty():
    assert _strip_foreign_scripts("你好世界") == ""


def test_streaming_tokens_fragmented_cjk():
    """Simulates qwen leaking one Chinese char per delta in a stream."""
    stream = ["o", "的", "目", "的", "地", "信", "息", "。"]
    joined = "".join(_strip_foreign_scripts(t) for t in stream)
    assert joined == "o"
