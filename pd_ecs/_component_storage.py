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
    """Chunked backing store for a single component.

    Data is kept in a list of fixed-size numpy arrays (chunks).  Only the last
    chunk may be partially filled, so adding entities never reallocates earlier
    chunks.  Sorting rewrites values in-place into the same chunk objects.
    """

    def __init__(self, dtype, chunk_size, name=None):
        self.name = name
        # dtype=None means infer from first data appended
        self._dtype = np.dtype(dtype) if dtype is not None else None
        self._chunk_size = int(chunk_size)
        self._data_chunks = []   # list of np.ndarray, each pre-allocated to chunk_size
        self._eid_chunks = []    # parallel list for entity ids (int64)
        self._last_fill = 0      # valid elements in the last chunk
        self._series_cache = None  # invalidated on any write

    @property
    def _length(self):
        if not self._data_chunks:
            return 0
        return (len(self._data_chunks) - 1) * self._chunk_size + self._last_fill

    def __len__(self):
        return self._length

    def _iter_valid(self):
        n = len(self._data_chunks)
        for i, (ec, dc) in enumerate(zip(self._eid_chunks, self._data_chunks)):
            fill = self._chunk_size if i < n - 1 else self._last_fill
            yield ec[:fill], dc[:fill]

    def all_eids(self):
        parts = [ec for ec, _ in self._iter_valid()]
        return np.concatenate(parts) if parts else np.array([], dtype=np.int64)

    def all_values(self):
        parts = [dc for _, dc in self._iter_valid()]
        if not parts:
            return np.array([], dtype=self._dtype if self._dtype is not None else np.int64)
        return np.concatenate(parts)

    @property
    def index(self):
        return pd.Index(self.all_eids())

    @property
    def values(self):
        return self.all_values()

    @property
    def loc(self):
        return _StorageLoc(self)

    def to_series(self):
        if self._series_cache is None:
            dtype = self._dtype if self._dtype is not None else np.int64
            self._series_cache = pd.Series(
                self.all_values(),
                index=pd.Index(self.all_eids()),
                name=self.name,
                dtype=dtype,
            )
        return self._series_cache

    def _invalidate(self):
        self._series_cache = None

    def _promote_dtype(self, new_dtype):
        """Upgrade all chunk arrays to new_dtype (mirrors pandas inplace upcast)."""
        new_dtype = np.dtype(new_dtype)
        self._data_chunks = [dc.astype(new_dtype) for dc in self._data_chunks]
        self._dtype = new_dtype
        self._invalidate()

    def append(self, eids, values):
        eids = np.asarray(eids, dtype=np.int64)
        raw = values.values if hasattr(values, 'values') else values
        values_arr = np.asarray(raw)
        if self._dtype is None:
            if len(values_arr):
                self._dtype = values_arr.dtype
        else:
            if not np.can_cast(values_arr.dtype, self._dtype, casting='safe'):
                self._promote_dtype(np.result_type(self._dtype, values_arr.dtype))
            values_arr = values_arr.astype(self._dtype)

        n = len(eids)
        if n == 0:
            return
        self._invalidate()
        pos = 0
        cs = self._chunk_size

        # Fill remaining space in last chunk
        if self._data_chunks and self._last_fill < cs:
            space = cs - self._last_fill
            take = min(space, n)
            self._data_chunks[-1][self._last_fill:self._last_fill + take] = values_arr[:take]
            self._eid_chunks[-1][self._last_fill:self._last_fill + take] = eids[:take]
            self._last_fill += take
            pos = take

        # Allocate new chunks for remaining data
        while pos < n:
            take = min(cs, n - pos)
            dc = np.empty(cs, dtype=self._dtype or np.int64)
            ec = np.empty(cs, dtype=np.int64)
            dc[:take] = values_arr[pos:pos + take]
            ec[:take] = eids[pos:pos + take]
            self._data_chunks.append(dc)
            self._eid_chunks.append(ec)
            self._last_fill = take
            pos += take

    def reorder_inplace(self, order):
        """Reorder elements by permutation, writing back into existing chunk memory."""
        if self._length == 0:
            return
        self._invalidate()
        new_data = self.all_values()[order]
        new_eids = self.all_eids()[order]
        pos = 0
        n = len(self._data_chunks)
        for i, (ec, dc) in enumerate(zip(self._eid_chunks, self._data_chunks)):
            fill = self._chunk_size if i < n - 1 else self._last_fill
            dc[:fill] = new_data[pos:pos + fill]
            ec[:fill] = new_eids[pos:pos + fill]
            pos += fill

    def delete_mask(self, bool_mask):
        """Return new storage with elements where bool_mask is True removed."""
        keep = ~np.asarray(bool_mask, dtype=bool)
        all_e = self.all_eids()
        all_v = self.all_values()
        new = ComponentStorage(self._dtype, self._chunk_size, self.name)
        if keep.any():
            new.append(all_e[keep], all_v[keep])
        return new

    def delete_eids(self, eids_to_delete):
        all_e = self.all_eids()
        return self.delete_mask(np.isin(all_e, eids_to_delete))

    def get_index_range(self, start, stop):
        """Return entity ids at logical positions [start, stop)."""
        cs = self._chunk_size
        parts = []
        pos = start
        while pos < stop:
            ci = pos // cs
            co = pos % cs
            chunk_end = min((ci + 1) * cs, stop)
            parts.append(self._eid_chunks[ci][co:chunk_end - ci * cs])
            pos = chunk_end
        return np.concatenate(parts) if parts else np.array([], dtype=np.int64)

    def get_slice_specs(self, start, stop):
        """Return list of (chunk_array, chunk_start, chunk_stop) for EntityView."""
        cs = self._chunk_size
        result = []
        pos = start
        while pos < stop:
            ci = pos // cs
            co = pos % cs
            chunk_end = min((ci + 1) * cs, stop)
            result.append((self._data_chunks[ci], co, chunk_end - ci * cs))
            pos = chunk_end
        return result

    def set_by_eids(self, eids, values):
        """Set values at specific entity ids (must already exist in storage)."""
        all_e = self.all_eids()
        positions = np.asarray(
            pd.Index(all_e).get_indexer(np.asarray(eids, dtype=np.int64)),
            dtype=np.intp,
        )
        if (positions < 0).any():
            raise KeyError("entity ids not found in component storage")
        if len(positions) == 0:
            return
        raw = values.values if isinstance(values, pd.Series) else values
        values_arr = np.asarray(raw)
        if self._dtype is not None:
            if not np.can_cast(values_arr.dtype, self._dtype, casting='safe'):
                self._promote_dtype(np.result_type(self._dtype, values_arr.dtype))
            values_arr = values_arr.astype(self._dtype)
        self._invalidate()
        cs = self._chunk_size
        ci_arr = positions // cs
        co_arr = positions % cs
        vals_flat = values_arr.ravel()

        # Sort by chunk index so writes to each chunk are one contiguous slice op.
        sort_ord = np.argsort(ci_arr, kind='stable')
        s_ci = ci_arr[sort_ord]
        s_co = co_arr[sort_ord]
        s_vals = vals_flat[sort_ord]
        unique_ci, first = np.unique(s_ci, return_index=True)
        ends = np.empty(len(unique_ci), dtype=np.intp)
        ends[:-1] = first[1:]
        ends[-1] = len(s_ci)
        for ci, a, b in zip(unique_ci.tolist(), first.tolist(), ends.tolist()):
            self._data_chunks[int(ci)][s_co[a:b]] = s_vals[a:b]
