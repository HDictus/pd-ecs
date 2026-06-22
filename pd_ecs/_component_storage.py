import numpy as np
import pandas as pd


class _StorageLoc:
    def __init__(self, storage):
        self._s = storage

    def __setitem__(self, key, values):
        if np.isscalar(key):
            eids = np.array([key], dtype=np.int64)
        else:
            eids = np.asarray(key, dtype=np.int64)
        vals = np.asarray(values.values if isinstance(values, pd.Series) else values)
        if vals.ndim == 0:
            vals = vals.reshape(1)
        self._s.set_by_eids(eids, vals)


class ComponentStorage:
    """Flat numpy backing store for a single component."""

    def __init__(self, dtype, chunk_size=None, name=None):
        self.name = name
        self._dtype = np.dtype(dtype) if dtype is not None else None
        self._eids = np.array([], dtype=np.int64)
        self._values = np.array([], dtype=self._dtype if self._dtype is not None else np.float64)

    @property
    def _length(self):
        return len(self._eids)

    @property
    def loc(self):
        return _StorageLoc(self)

    def all_eids(self):
        return self._eids

    def all_values(self):
        return self._values

    def to_series(self):
        dtype = self._dtype if self._dtype is not None else np.int64
        return pd.Series(self._values, index=pd.Index(self._eids), name=self.name, dtype=dtype)

    def _promote_dtype(self, new_dtype):
        new_dtype = np.dtype(new_dtype)
        self._values = self._values.astype(new_dtype)
        self._dtype = new_dtype

    def append(self, eids, values):
        eids = np.asarray(eids, dtype=np.int64)
        raw = values.values if hasattr(values, 'values') else values
        values_arr = np.asarray(raw)

        if len(eids) == 0:
            return

        if self._dtype is None:
            self._dtype = values_arr.dtype
            self._values = self._values.astype(self._dtype)
        elif not np.can_cast(values_arr.dtype, self._dtype, casting='safe'):
            self._promote_dtype(np.result_type(self._dtype, values_arr.dtype))

        values_arr = values_arr.astype(self._dtype)
        self._eids = np.concatenate([self._eids, eids])
        self._values = np.concatenate([self._values, values_arr])

    def reorder_inplace(self, order):
        order = np.asarray(order)
        self._eids = self._eids[order]
        self._values = self._values[order]

    def delete_mask(self, bool_mask):
        keep = ~np.asarray(bool_mask, dtype=bool)
        new = ComponentStorage(self._dtype, name=self.name)
        new._eids = self._eids[keep]
        new._values = self._values[keep]
        return new

    def delete_eids(self, eids_to_delete):
        return self.delete_mask(np.isin(self._eids, eids_to_delete))

    def get_index_range(self, start, stop):
        return self._eids[start:stop]

    def get_slice_specs(self, start, stop):
        return [(self._values, start, stop)]

    def set_by_eids(self, eids, values):
        eids = np.asarray(eids, dtype=np.int64)
        raw = values.values if isinstance(values, pd.Series) else values
        values_arr = np.asarray(raw)
        if len(values_arr) == 0:
            return
        if self._dtype is not None:
            if not np.can_cast(values_arr.dtype, self._dtype, casting='safe'):
                self._promote_dtype(np.result_type(self._dtype, values_arr.dtype))
            values_arr = values_arr.astype(self._dtype)
        positions = pd.Index(self._eids).get_indexer(eids)
        if (positions < 0).any():
            raise KeyError("entity ids not found in component storage")
        self._values[positions] = values_arr
