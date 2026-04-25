import sys
import pytest
sys.path.insert(0, "benchmarks")
from bench_anticipate_signals import bench_signal, bench_all, render_text, _percentile


def test_percentile_basic():
    assert _percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == pytest.approx(5.5)
    assert _percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == pytest.approx(9.55, abs=0.1)
    assert _percentile([], 50) == 0.0


def test_bench_signal_returns_expected_keys():
    def fake_signal(now):
        return []
    result = bench_signal("test", fake_signal, iterations=3)
    assert "cold_ms" in result
    assert "warm_p50_ms" in result
    assert "warm_p95_ms" in result
    assert "emit_counts" in result
    assert "emit_total" in result
    assert result["emit_total"] == 0


def test_bench_signal_handles_exception():
    def crash_signal(now):
        raise RuntimeError("boom")
    result = bench_signal("crash", crash_signal, iterations=3)
    assert "error" in result


def test_bench_signal_emit_count():
    from rag import AnticipatoryCandidate
    def emitting(now):
        return [AnticipatoryCandidate(kind="x", score=0.5, message="m", dedup_key="k", snooze_hours=2, reason="r")]
    result = bench_signal("emit", emitting, iterations=3)
    assert result["emit_total"] == 3  # 3 iterations × 1 emit


def test_bench_signal_warm_iterations():
    def fast(now):
        return []
    result = bench_signal("fast", fast, iterations=5)
    # cold + 4 warm
    assert len(result["emit_counts"]) == 5


def test_render_text_includes_signal_names():
    bench = {
        "iterations": 3,
        "signals": {
            "calendar": {"cold_ms": 10, "warm_p50_ms": 5, "warm_p95_ms": 7, "emit_total": 1, "emit_counts": [1, 1, 1]},
        },
        "total_cold_ms": None,
        "total_warm_p50_ms": None,
    }
    text = render_text(bench)
    assert "calendar" in text
    assert "cold=10" in text


def test_bench_all_with_signal_filter(monkeypatch):
    """bench_all con --signal solo corre esa signal."""
    import rag
    def fake_sig(now): return []
    fake_signals = (("calendar", fake_sig), ("echo", fake_sig))
    monkeypatch.setattr(rag, "_ANTICIPATE_SIGNALS", fake_signals)
    result = bench_all(iterations=2, signal_filter="calendar")
    assert "calendar" in result["signals"]
    assert "echo" not in result["signals"]
