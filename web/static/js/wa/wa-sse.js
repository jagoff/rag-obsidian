// EventSource wrapper para `/api/wa/stream`. Maneja reconnect
// automático (EventSource lo hace nativo) + dispatch tipado a callbacks.

const handlers = {
  new_message: [],
  chat_update: [],
  reaction_changed: [],
  message_revoked: [],
  presence: [],
  hello: [],
  heartbeat: [],
};

let es = null;
let opened = false;
let onStateChangeCallback = null;

function dispatch(name, payload) {
  const list = handlers[name];
  if (!list) return;
  for (const fn of list) {
    try {
      fn(payload);
    } catch (e) {
      console.error(`[wa-sse] handler ${name} crashed`, e);
    }
  }
}

export function on(name, fn) {
  if (!handlers[name]) throw new Error(`unknown event: ${name}`);
  handlers[name].push(fn);
}

export function connect() {
  if (es) return;
  es = new EventSource("/api/wa/stream");

  for (const name of Object.keys(handlers)) {
    es.addEventListener(name, (ev) => {
      let data = {};
      try {
        data = ev.data ? JSON.parse(ev.data) : {};
      } catch {
        data = { raw: ev.data };
      }
      dispatch(name, data);
    });
  }

  es.addEventListener("open", () => {
    opened = true;
    if (onStateChangeCallback) onStateChangeCallback(true);
  });

  es.addEventListener("error", () => {
    // EventSource auto-reconnect; solo refleja el estado a la UI.
    if (opened) {
      opened = false;
      if (onStateChangeCallback) onStateChangeCallback(false);
    }
  });
}

export function onConnectionState(fn) {
  onStateChangeCallback = fn;
}

export function isOpen() {
  return opened;
}
