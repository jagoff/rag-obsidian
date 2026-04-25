"""Signals sub-package. Cada signal en su propio módulo `<kind>.py`.

Los módulos se auto-importan desde `rag_anticipate/__init__.py::
_autodiscover_signals`. NO editar shared state acá — usar el decorator
`@register_signal` desde `rag_anticipate.signals.base`.
"""
