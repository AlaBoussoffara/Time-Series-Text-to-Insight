# datastore.py
from __future__ import annotations
from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import threading, time, uuid

@dataclass
class Entry:
    df: pd.DataFrame
    created_at: float
    ttl_s: Optional[int]

class DataStore:
    def __init__(self, gc_interval_s: int = 30, max_items: Optional[int] = 5000):
        self._lock = threading.RLock()
        self._store: Dict[str, Entry] = {}
        self._gc_interval_s = gc_interval_s
        self._max_items = max_items
        self._stop = threading.Event()
        self._gc = threading.Thread(target=self._gc_loop, daemon=True); self._gc.start()

    def put_df(self, df: pd.DataFrame, namespace="result", ttl_s: Optional[int]=3600) -> str:
        ref = f"{namespace}/{uuid.uuid4().hex}"
        with self._lock:
            if self._max_items and len(self._store) >= self._max_items:
                self._evict_oldest()
            self._store[ref] = Entry(df.copy(), time.time(), ttl_s)
        return ref

    def get_df(self, ref: str) -> pd.DataFrame:
        with self._lock:
            e = self._store.get(ref)
            if e is None: raise KeyError(f"df_ref inconnu: {ref}")
            return e.df.copy()

    def _expired(self, e: Entry) -> bool:
        return e.ttl_s is not None and (time.time()-e.created_at) > e.ttl_s

    def _gc_loop(self):
        while not self._stop.is_set():
            time.sleep(self._gc_interval_s)
            with self._lock:
                for ref in [r for r,e in self._store.items() if self._expired(e)]:
                    self._store.pop(ref, None)

    def _evict_oldest(self):
        if not self._store: return
        oldest = min(self._store.items(), key=lambda kv: kv[1].created_at)[0]
        self._store.pop(oldest, None)

DATASTORE = DataStore()

def store_df(df: pd.DataFrame, namespace="result", ttl_s: Optional[int]=3600) -> str:
    return DATASTORE.put_df(df, namespace, ttl_s)

def load_df(ref: str) -> pd.DataFrame:
    return DATASTORE.get_df(ref)
