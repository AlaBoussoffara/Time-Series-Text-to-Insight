from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd
import threading, time, uuid

class DataStore:
    """
    _store: Dict[
      ref: str,
      {
        "description": str,
        "df": pd.DataFrame,
        "created_at": float,
        "ttl": Optional[int]
      }
    ]
    """
    def __init__(self, gc_interval_s: int = 30, max_items: Optional[int] = 10000):
        self._lock = threading.RLock()
        self._store: Dict[str, Dict[str, Any]] = {}
        self._gc_interval_s = gc_interval_s
        self._max_items = max_items
        self._stop = threading.Event()
        self._gc_thread = threading.Thread(target=self._gc_loop, daemon=True)
        self._gc_thread.start()

    # ------------ API ------------
    def put(
        self,
        df: pd.DataFrame,
        *,
        description: str = "",
        ttl: Optional[int] = 3600,
        namespace: str = "result",
        ref: Optional[str] = None,
        upsert: bool = False,
    ) -> str:
        """
        Stocke un DF. Si 'ref' est fourni par l'agent, on l'utilise tel quel.
        Sinon on génère: f"{namespace}/{uuid4}".
        upsert=False => erreur si la ref existe déjà.
        """
        if ref is None:
            ref = f"{namespace}/{uuid.uuid4().hex}"

        entry = {
            "description": description or self._auto_description(df),
            "df": df.copy(),
            "created_at": time.time(),
            "ttl": ttl,
        }
        with self._lock:
            if (not upsert) and ref in self._store:
                raise ValueError(f"Ref déjà présente: {ref}")
            if self._max_items and len(self._store) >= self._max_items and (ref not in self._store):
                self._evict_oldest()
            self._store[ref] = entry
        return ref

    def get_df(self, ref: str) -> pd.DataFrame:
        with self._lock:
            e = self._require(ref)
            return e["df"].copy()

    def get_description(self, ref: str) -> str:
        with self._lock:
            e = self._require(ref)
            return str(e["description"])

    def set_description(self, ref: str, description: str) -> None:
        with self._lock:
            e = self._require(ref)
            e["description"] = str(description)

    def delete(self, ref: str) -> None:
        with self._lock:
            self._store.pop(ref, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def exists(self, ref: str) -> bool:
        with self._lock:
            return ref in self._store

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {"items": len(self._store),
                    "namespaces": sorted({r.split("/")[0] for r in self._store})}

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Return lightweight metadata for all stored datasets."""
        with self._lock:
            snapshot: Dict[str, Dict[str, Any]] = {}
            for ref, entry in self._store.items():
                df = entry.get("df")
                row_count = len(df) if isinstance(df, pd.DataFrame) else 0
                snapshot[ref] = {
                    "description": entry.get("description", ""),
                    "row_count": row_count,
                    "datastore_ref": ref,
                }
            return snapshot

    # ------------ internes ------------
    def _require(self, ref: str) -> Dict[str, Any]:
        e = self._store.get(ref)
        if e is None:
            raise KeyError(f"df_ref inconnu ou expiré: {ref}")
        return e

    def _expired(self, e: Dict[str, Any]) -> bool:
        ttl = e.get("ttl")
        return False if ttl is None else (time.time() - e["created_at"]) > ttl

    def _gc_loop(self):
        while not self._stop.is_set():
            time.sleep(self._gc_interval_s)
            with self._lock:
                for r in [r for r, e in self._store.items() if self._expired(e)]:
                    self._store.pop(r, None)

    def _evict_oldest(self):
        if not self._store:
            return
        oldest_ref = min(self._store.items(), key=lambda kv: kv[1]["created_at"])[0]
        self._store.pop(oldest_ref, None)

    # petite description auto si rien n'est fourni
    def _auto_description(self, df: pd.DataFrame) -> str:
        cols = ", ".join(map(str, df.columns.tolist()))
        desc = f"Lignes: {len(df)} | Colonnes: {cols}"
        if "ts" in df.columns:
            try:
                ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
                if ts.notna().any():
                    desc += f" | Intervalle ts: {ts.min()} → {ts.max()}"
            except Exception:
                pass
        return desc


# -------- Singleton + helpers --------
DATASTORE = DataStore()

def store_df(df: pd.DataFrame, *, description: str = "", ttl: Optional[int] = 3600,
             namespace: str = "result", ref: Optional[str] = None, upsert: bool = False) -> str:
    return DATASTORE.put(df, description=description, ttl=ttl, namespace=namespace, ref=ref, upsert=upsert)

def load_df(ref: str) -> pd.DataFrame:
    return DATASTORE.get_df(ref)

def load_description(ref: str) -> str:
    return DATASTORE.get_description(ref)

def save_description(ref: str, description: str) -> None:
    DATASTORE.set_description(ref, description)
