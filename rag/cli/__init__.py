"""Sub-package que aloja los grupos / subcomandos Click extraídos del
monolito histórico ``rag/__init__.py``.

Diseño del split (ver ``plans/package-split-2026-04-29.md``):

* El **grupo raíz** ``cli`` sigue viviendo en ``rag/__init__.py`` — es el
  registry contra el que cada submódulo registra sus subcomandos. Movernos
  el grupo raíz acá generaría import circular (``rag/__init__.py`` lo usa
  para wirear flags globales en cada subcomando legacy todavía no
  extraído).
* Cada submódulo de ``rag.cli.*`` define **un** grupo Click + sus
  subcomandos directos y, al importarse, lo registra contra el ``cli``
  raíz vía decorador ``@cli.group()`` (o ``cli.add_command(...)`` cuando
  hace falta resolver circulares).
* Para que la registración ocurra simplemente con ``import rag``, este
  ``__init__.py`` re-importa todos los submódulos al final del init de
  ``rag/__init__.py`` (después de que el grupo raíz y los helpers están
  definidos). Ver el bloque "CLI sub-package re-export shim" al pie del
  monolito.

Este sub-package es **piloto** del patrón (Sub-chunk 1.1 del plan).
Empieza con un único submódulo (``vault``) para validar la mecánica
antes de migrar grupos más grandes (``query``, ``chat``, ``productivity``).
"""

# El barrel import vive en ``rag/__init__.py`` (al final del archivo, post
# definición del grupo raíz ``cli`` y sus helpers compartidos). Ver el
# comentario en la sección "CLI sub-package re-export shim" del monolito.
#
# Mantenemos este ``__init__`` *vacío* en runtime para evitar
# trigger-imports prematuros cuando alguien hace ``from rag.cli import X``
# antes de que ``rag`` haya terminado de inicializarse.
