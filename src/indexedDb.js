/* ---------- Session storage (IndexedDB) ----------
 * Persists recordings + settings as named projects — originally just a
 * single crash-recovery slot ("did the tab close before I finished?"), now
 * a small library so someone working on more than one song idea doesn't
 * have each new recording silently overwrite the last. Every export
 * resolves to a harmless no-op result on failure (e.g. private-browsing
 * IndexedDB restrictions) rather than throwing, since this is a nice-to-
 * have and must never break the app around it.
 */

const DB_NAME = 'stamma';
const DB_VERSION = 2;
const STORE_NAME = 'projects';
const OLD_STORE_NAME = 'session'; // v1's single fixed-key store, migrated below
const OLD_SESSION_KEY = 'current';

function generateId() {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function openDb() {
  return new Promise((resolve, reject) => {
    if (!window.indexedDB) {
      reject(new Error('IndexedDB not available'));
      return;
    }
    let req;
    try {
      req = window.indexedDB.open(DB_NAME, DB_VERSION);
    } catch (e) {
      reject(e);
      return;
    }
    req.onupgradeneeded = (event) => {
      const db = req.result;
      const tx = req.transaction;
      const store = db.objectStoreNames.contains(STORE_NAME)
        ? tx.objectStore(STORE_NAME)
        : db.createObjectStore(STORE_NAME, { keyPath: 'id' });
      // v1 -> v2: the old version kept exactly one autosave under a fixed
      // key — carry it over as a first named project rather than losing
      // whatever someone had in progress when this shipped.
      if (event.oldVersion < 2 && db.objectStoreNames.contains(OLD_STORE_NAME)) {
        const oldStore = tx.objectStore(OLD_STORE_NAME);
        const getReq = oldStore.get(OLD_SESSION_KEY);
        getReq.onsuccess = () => {
          const old = getReq.result;
          if (old) {
            store.put({ ...old, id: generateId(), name: 'Återställd session' });
          }
          db.deleteObjectStore(OLD_STORE_NAME);
        };
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

// Creates a new project (no `id`) or overwrites an existing one (with
// `id`) — returns the id either way, so the caller can keep autosaving
// into the same project on subsequent calls.
export async function saveProject(project) {
  const db = await openDb();
  try {
    const id = project.id || generateId();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).put({ ...project, id });
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
    return id;
  } finally {
    db.close();
  }
}

// Metadata only (id, name, savedAt, ...settings), newest first — strips
// the audio blob so browsing the list doesn't have to transfer every
// project's WAV data just to show a name and a date.
export async function listProjects() {
  const db = await openDb();
  try {
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const req = tx.objectStore(STORE_NAME).getAll();
      req.onsuccess = () => {
        const all = (req.result || []).map(({ audio: _audio, ...meta }) => meta);
        all.sort((a, b) => b.savedAt - a.savedAt);
        resolve(all);
      };
      req.onerror = () => reject(req.error);
    });
  } finally {
    db.close();
  }
}

export async function loadProject(id) {
  const db = await openDb();
  try {
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const req = tx.objectStore(STORE_NAME).get(id);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => reject(req.error);
    });
  } finally {
    db.close();
  }
}

export async function deleteProject(id) {
  const db = await openDb();
  try {
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).delete(id);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } finally {
    db.close();
  }
}

export async function renameProject(id, name) {
  const db = await openDb();
  try {
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const getReq = store.get(id);
      getReq.onsuccess = () => {
        if (getReq.result) store.put({ ...getReq.result, name });
      };
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } finally {
    db.close();
  }
}
