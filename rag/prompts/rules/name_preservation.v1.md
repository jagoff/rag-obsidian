---
name: name_preservation
version: v1
date: 2026-04-21
kind: rule
notes: |
  Name-preservation clause — prepended right after chunk_as_data on every
  SYSTEM_RULES* so the model NEVER 'corrects' proper nouns it doesn't recognise.
  Motivation (2026-04-21): a user asked about "Bizarrap" (Argentine music
  producer), the vault had no musical info, and the LLM answered refusing about
  "Bizarra" — silently swapping the proper noun to a more common dictionary
  word. Unacceptable: proper nouns are the user's ground truth, not the
  model's.
---
REGLA DE NOMBRES PROPIOS: si el usuario menciona un nombre propio, marca, artista, persona, lugar, producto o término técnico en su pregunta, copialo TEXTUAL como aparece escrito por el usuario. NUNCA lo 'corrijas' aunque te parezca raro, mal escrito o desconocido. Ejemplo: si pregunta por 'Bizarrap', NO escribas 'Bizarra' ni 'Vizarrap' ni ninguna variante — solo 'Bizarrap'. Si no reconocés el término, tratalo como un nombre propio válido que el usuario conoce y vos no.

