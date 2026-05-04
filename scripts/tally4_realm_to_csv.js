#!/usr/bin/env node
/*
 * tally4_realm_to_csv.js — extrae transacciones de un `moze.realm` (DB
 * propietaria de la app Tally4 / MOZE) a un CSV con la misma shape que
 * los `MOZE_*.csv` históricos, así el resto del pipeline de
 * obsidian-rag (`_fetch_finance`, `_sync_moze_notes`, dashboard) sigue
 * consumiendo CSV sin enterarse del cambio de formato.
 *
 * USO
 *   node tally4_realm_to_csv.js --realm <path-al-moze.realm> [--out <csv>]
 *
 * Salida: si `--out` se omite, el CSV va a stdout.
 *
 * SHAPE del CSV (16 columnas, mismo header que las exports históricas):
 *   Account,Currency,Type,Main Category,Subcategory,Price,Fee,Bonus,
 *   Name,Store,Date,Time,Project,Note,Tags,Target
 *
 * Type: enum entero en realm → string. Mapeo observado en MOZE 4.0:
 *   0 → Expense, 1 → Income, 3 → Receivable, 7 → Balance Adjustment.
 *   Transfers viven en `AHTransfer` (no `AHRecord`) y NO se exportan
 *   acá — el caso real del user no los usa todavía.
 *
 * Date / Time: realm guarda `date` como UTC ISO + `timeSeconds` como
 *   segundos del día en la TZ del device. Para mantener compat con el
 *   CSV histórico (donde Date era MM/DD/YYYY local y Time HH:MM:ss
 *   local), partimos de `dateString` (`YYYY.MM.DD-HH:mm:ss` ya en TZ
 *   local) cuando está, y caemos a `date` UTC si falta.
 *
 * Filtros: skipea `isDeleted=true` y `isEnabled=false`.
 *
 * IMPORTANTE: el archivo `.realm` se abre WRITABLE para que realm-js
 * pueda auto-upgradeear el formato (Tally4 4.0 escribe formato v22 y
 * realm 12+ se queja del read-only). El caller debería trabajar con
 * una COPIA del realm en tmpdir, no el original en iCloud.
 */

const Realm = require("realm");
const fs = require("fs");
const path = require("path");

const args = process.argv.slice(2);
let realmPath = null;
let outPath = null;
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--realm" && args[i + 1]) {
    realmPath = args[++i];
  } else if (args[i] === "--out" && args[i + 1]) {
    outPath = args[++i];
  } else if (args[i] === "--help" || args[i] === "-h") {
    console.error("Usage: node tally4_realm_to_csv.js --realm <path> [--out <csv>]");
    process.exit(0);
  }
}
if (!realmPath) {
  console.error("ERROR: --realm <path> is required");
  process.exit(2);
}
realmPath = path.resolve(realmPath);
if (!fs.existsSync(realmPath)) {
  console.error(`ERROR: realm file not found: ${realmPath}`);
  process.exit(2);
}

// ── Helpers ──────────────────────────────────────────────────────────────

const TYPE_MAP = {
  0: "Expense",
  1: "Income",
  3: "Receivable",
  7: "Balance Adjustment",
};

function csvEscape(v) {
  if (v == null) return "";
  let s = String(v);
  if (s.includes(",") || s.includes("\"") || s.includes("\n") || s.includes("\r")) {
    s = "\"" + s.replace(/"/g, '""') + "\"";
  }
  return s;
}

function fmtPriceES(n) {
  // CSVs históricos: ES decimals (`-15000`, `2026,74`, `1.234,56`).
  // Mantenemos coma decimal sin separador de miles para parseo simple
  // (el parser Python `_moze_pnum` quita puntos primero, después
  // transforma coma→punto, así que sin puntos no rompe).
  if (typeof n !== "number" || !isFinite(n)) return "";
  // Round to 4 decimals max y trim ceros.
  const rounded = Math.round(n * 10000) / 10000;
  return String(rounded).replace(".", ",");
}

function dateFromString(dateString, fallbackDate) {
  // dateString: "YYYY.MM.DD-HH:mm:ss" (TZ local).
  // Returns { dateMDY: "MM/DD/YYYY", time: "HH:MM:ss" } or null.
  if (typeof dateString === "string" && /^\d{4}\.\d{2}\.\d{2}-\d{2}:\d{2}:\d{2}/.test(dateString)) {
    const [datePart, timePart] = dateString.split("-");
    const [y, mo, d] = datePart.split(".");
    return { dateMDY: `${mo}/${d}/${y}`, time: timePart };
  }
  // Fallback: usar UTC date como MDY + HH:mm derivado de timeSeconds.
  if (fallbackDate instanceof Date && !isNaN(fallbackDate.getTime())) {
    const y = fallbackDate.getUTCFullYear();
    const mo = String(fallbackDate.getUTCMonth() + 1).padStart(2, "0");
    const d = String(fallbackDate.getUTCDate()).padStart(2, "0");
    const hh = String(fallbackDate.getUTCHours()).padStart(2, "0");
    const mm = String(fallbackDate.getUTCMinutes()).padStart(2, "0");
    const ss = String(fallbackDate.getUTCSeconds()).padStart(2, "0");
    return { dateMDY: `${mo}/${d}/${y}`, time: `${hh}:${mm}:${ss}` };
  }
  return null;
}

function safeName(linked) {
  if (!linked) return "";
  try {
    return linked.name || "";
  } catch (_) {
    return "";
  }
}

function safeCode(linked) {
  if (!linked) return "";
  try {
    return linked.code || "";
  } catch (_) {
    return "";
  }
}

function categoryName(classification) {
  // AHClassification.category → AHCategory (linked). MOZE CSV tenía
  // "Main Category" (categoría) y "Subcategory" (classification). Si
  // el record carece de classification.category, dejamos Main vacío.
  if (!classification) return "";
  try {
    return classification.category ? classification.category.name || "" : "";
  } catch (_) {
    return "";
  }
}

// ── Apertura del realm ───────────────────────────────────────────────────

let realm;
try {
  realm = new Realm({ path: realmPath });
} catch (e) {
  console.error(`ERROR: cannot open realm: ${e.message}`);
  process.exit(3);
}

const records = realm.objects("AHRecord").sorted("date", false);

const HEADER = [
  "Account", "Currency", "Type", "Main Category", "Subcategory",
  "Price", "Fee", "Bonus", "Name", "Store", "Date", "Time",
  "Project", "Note", "Tags", "Target",
];

const lines = [HEADER.join(",")];
let kept = 0;
let skipped = 0;

for (const r of records) {
  if (r.isDeleted || r.isEnabled === false) {
    skipped++;
    continue;
  }
  const typeStr = TYPE_MAP[r.type];
  if (!typeStr) {
    // Tipos exóticos (transfers, eventos compuestos) no caben en la
    // shape MOZE — saltar pero contar.
    skipped++;
    continue;
  }
  const datePair = dateFromString(r.dateString, r.date);
  if (!datePair) {
    skipped++;
    continue;
  }
  const project = safeName(r.project);
  // El sentinel `NO_PROJECT` que usa Tally4 internamente NO sale al CSV.
  const projectOut = project === "NO_PROJECT" ? "" : project;
  const target = safeName(r.target);

  const row = [
    csvEscape(safeName(r.account)),
    csvEscape(safeCode(r.currency)),
    csvEscape(typeStr),
    csvEscape(categoryName(r.classification)),
    csvEscape(safeName(r.classification)),
    csvEscape(fmtPriceES(r.price)),
    csvEscape(fmtPriceES(r.fee)),
    csvEscape(fmtPriceES(r.bonus)),
    csvEscape(r.name || ""),
    csvEscape(r.store || ""),
    csvEscape(datePair.dateMDY),
    csvEscape(datePair.time),
    csvEscape(projectOut),
    csvEscape(r.desc || ""),
    csvEscape(r.tags || ""),
    csvEscape(target),
  ];
  lines.push(row.join(","));
  kept++;
}

realm.close();

const out = lines.join("\n") + "\n";
if (outPath) {
  fs.writeFileSync(path.resolve(outPath), out, "utf8");
  console.error(`tally4_realm_to_csv: kept=${kept} skipped=${skipped} → ${outPath}`);
} else {
  process.stdout.write(out);
  console.error(`tally4_realm_to_csv: kept=${kept} skipped=${skipped}`);
}

// realm-js mantiene workers/listeners abiertos que impiden el natural exit
// del event loop incluso después de `realm.close()`. Forzamos exit explícito
// para que `subprocess.run(..., capture_output=True)` no se cuelgue para
// siempre esperando que cierren los pipes.
process.exit(0);
