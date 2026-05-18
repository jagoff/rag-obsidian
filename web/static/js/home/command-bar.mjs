// command-bar.mjs — KPIs prominentes con trend en la command bar superior.

import { isActionableWhatsApp, isReminderDueToday, setKPI } from "./core.mjs";

export function renderCmdBar(payload) {
  const inboxToday = payload.today?.evidence?.inbox_today || [];
  const reminders = payload.signals?.reminders || [];
  const wa = payload.signals?.whatsapp_unreplied || [];
  const actionWa = wa.filter(isActionableWhatsApp);
  const loops = payload.signals?.loops_stale || [];

  const showKPI = (id, visible, data) => {
    setKPI(id, data);
    const el = document.getElementById(id);
    if (el) el.hidden = !visible;
    return visible ? 1 : 0;
  };

  let visibleCount = 0;

  const inboxCount = inboxToday.length;
  visibleCount += showKPI("kpi-inbox", inboxCount > 0, {
    value: inboxCount,
    tone: inboxCount === 0 ? "ok" : inboxCount > 5 ? "critical" : "warning",
    meta: inboxCount === 0 ? "todo procesado" :
          inboxCount === 1 ? "1 pendiente" :
          `${inboxCount} pendientes`,
  });

  const remindersDue = reminders.filter((r) => isReminderDueToday(r)).length;
  visibleCount += showKPI("kpi-reminders", remindersDue > 0, {
    value: remindersDue,
    tone: remindersDue === 0 ? "ok" : remindersDue > 3 ? "critical" : "warning",
    meta: remindersDue === 0 ? "sin reminders para hoy" :
          remindersDue === 1 ? "1 para hoy" :
          `${remindersDue} para hoy`,
  });

  visibleCount += showKPI("kpi-wa", actionWa.length > 0, {
    value: actionWa.length,
    tone: actionWa.length === 0 ? "ok" : actionWa.length > 8 ? "critical" : "warning",
    meta: actionWa.length === 0 ? "todo respondido" :
          actionWa.length === 1 ? "1 chat requiere mirar" :
          `${actionWa.length} chats requieren mirar`,
  });

  visibleCount += showKPI("kpi-loops", loops.length > 0, {
    value: loops.length,
    tone: loops.length === 0 ? "ok" : loops.length > 5 ? "critical" : "warning",
    meta: loops.length === 0 ? "ningún loop envejeciendo" :
          loops.length === 1 ? "1 loop STALE" :
          `${loops.length} loops STALE`,
  });

  const cmdbar = document.querySelector(".cmdbar");
  if (cmdbar) cmdbar.hidden = visibleCount === 0;
}
