// command-bar.mjs — 4 KPIs prominentes con trend en la command bar superior.

import { setKPI } from "./core.mjs";

export function renderCmdBar(payload) {
  const inboxToday = payload.today?.evidence?.inbox_today || [];
  const reminders = payload.signals?.reminders || [];
  const wa = payload.signals?.whatsapp_unreplied || [];
  const loops = payload.signals?.loops_stale || [];

  const inboxCount = inboxToday.length;
  setKPI("kpi-inbox", {
    value: inboxCount,
    tone: inboxCount === 0 ? "ok" : inboxCount > 5 ? "critical" : "warning",
    meta: inboxCount === 0 ? "todo procesado" :
          inboxCount === 1 ? "1 pendiente" :
          `${inboxCount} pendientes`,
  });

  const remindersDue = reminders.filter((r) => {
    if (!r.due_date) return true;
    const due = new Date(r.due_date);
    const now = new Date();
    const eod = new Date(now); eod.setHours(23, 59, 59, 999);
    return due <= eod;
  }).length;
  setKPI("kpi-reminders", {
    value: remindersDue,
    tone: remindersDue === 0 ? "ok" : remindersDue > 3 ? "critical" : "warning",
    meta: remindersDue === 0 ? "tranquilo" :
          remindersDue === 1 ? "1 para hoy" :
          `${remindersDue} para hoy`,
  });

  setKPI("kpi-wa", {
    value: wa.length,
    tone: wa.length === 0 ? "ok" : wa.length > 8 ? "critical" : "warning",
    meta: wa.length === 0 ? "todo respondido" :
          wa.length === 1 ? "1 chat espera respuesta" :
          `${wa.length} chats esperan respuesta`,
  });

  setKPI("kpi-loops", {
    value: loops.length,
    tone: loops.length === 0 ? "ok" : loops.length > 5 ? "critical" : "warning",
    meta: loops.length === 0 ? "ningún loop envejeciendo" :
          loops.length === 1 ? "1 loop STALE" :
          `${loops.length} loops STALE`,
  });

  // Ocultar la cmdbar entera si todos los KPIs quedaron en 0 (sin items visibles).
  const cmdbar = document.querySelector(".cmdbar");
  if (cmdbar) {
    const visibleKpis = cmdbar.querySelectorAll(".kpi:not([hidden])");
    cmdbar.hidden = visibleKpis.length === 0;
  }
}
