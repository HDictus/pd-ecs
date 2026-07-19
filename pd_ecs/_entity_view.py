import numpy as np
import pandas as pd


def _write_to_chunks(slices, pos_in_view, values):
    """Write values at view-relative positions into chunk backing arrays.

    slices:       list of (chunk_array, chunk_start, chunk_stop)
    pos_in_view:  int array of positions within the view
    values:       scalar or 1-D array aligned with pos_in_view
    """
    if not slices:
        return
    pos = np.asarray(pos_in_view, dtype=np.intp)
    slice_lens = np.array([e - s for _, s, e in slices], dtype=np.intp)
    view_starts = np.empty(len(slices), dtype=np.intp)
    view_starts[0] = 0
    if len(slices) > 1:
        view_starts[1:] = np.cumsum(slice_lens[:-1])

    si_per = np.searchsorted(view_starts, pos, side='right') - 1
    vals = np.asarray(values)

    for si, (chunk_arr, chunk_start, _) in enumerate(slices):
        mask = si_per == si
        if not mask.any():
            continue
        offs = pos[mask] - view_starts[si]
        if vals.ndim == 0:
            chunk_arr[chunk_start + offs] = vals
        else:
            chunk_arr[chunk_start + offs] = vals[mask]


class EntityView:
    """
    Lazy view over entities matching a multi-component filter.

    .index is computed eagerly; component columns are materialized on demand
    by slicing the backing numpy arrays of each component's Series.
    """

    # TODO: should just accept arrays, not slices
    # what a mess I'm left to clean up
    def __init__(self, index, slices):
        """
        index:  pd.Index of matching entity IDs
        slices: dict[Component, list[(np.ndarray, int, int)]]
                Each entry is a (backing_array, start, stop) describing one
                contiguous archetype-run within the component's sorted Series.
        """
        self.index = index
        self._slices = slices

    @property
    def columns(self):
        return list(self._slices)

    def __contains__(self, comp):
        return comp in self._slices

    def __getitem__(self, key):
        if isinstance(key, list):
            return pd.DataFrame({comp: self[comp] for comp in key}, index=self.index)
        if isinstance(key, (pd.Series, np.ndarray)) and pd.api.types.is_bool_dtype(key):
            df = pd.DataFrame({comp: self[comp] for comp in self._slices}, index=self.index)
            return df[key]
        chunks = self._slices[key]
        if not chunks:
            return pd.Series([], index=self.index, name=key, dtype=float)
        if len(chunks) == 1:
            backing, start, stop = chunks[0]
            data = backing[start:stop]          # numpy view, no copy
        else:
            data = np.concatenate([b[s:e] for b, s, e in chunks])
        return pd.Series(data, index=self.index, name=key)

    def __setitem__(self, key, values):
        if isinstance(key, list):
            for comp in key:
                self[comp] = values[comp]
            return
        chunks = self._slices[key]
        flat = np.asarray(values.values if isinstance(values, pd.Series) else values)
        pos = 0
        for backing, start, stop in chunks:
            n = stop - start
            backing[start:stop] = flat[pos:pos + n]
            pos += n

    def to_frame(self):
        """Materialize all columns as a DataFrame."""
        return pd.DataFrame({comp: self[comp] for comp in self._slices}, index=self.index)

    def __getattr__(self, name):
        # Only reached when normal lookup fails; guard against pre-init access.
        if '_slices' not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.to_frame(), name)

    @property
    def loc(self):
        return _EntityViewLoc(self)

    def __len__(self):
        return len(self.index)


class _EntityViewLoc:
    def __init__(self, view):
        self._view = view

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key
            if isinstance(col_key, list):
                df = self._view[col_key]        # materialise selected columns
                return df.loc[row_key]
            series = self._view[col_key]        # materialise one column
            return series.loc[row_key]

        # row-only selection
        if pd.api.types.is_scalar(key):
            pos = self._view.index.get_loc(key)
            return pd.Series(
                {comp: self._view[comp].iloc[pos] for comp in self._view._slices},
                name=key,
            )
        data = {comp: self._view[comp].loc[key] for comp in self._view._slices}
        return pd.DataFrame(
            data,
            index=pd.Index(key),
        )

    def __setitem__(self, key, values):
        if not isinstance(key, tuple):
            raise ValueError("loc assignment requires (row_key, col_key)")
        row_key, col_key = key
        cols = col_key if isinstance(col_key, list) else [col_key]

        if pd.api.types.is_scalar(row_key):
            pos_in_view = np.array([self._view.index.get_loc(row_key)], dtype=np.intp)
        else:
            pos_in_view = np.asarray(self._view.index.get_indexer(row_key), dtype=np.intp)

        for i, comp in enumerate(cols):
            if isinstance(values, pd.DataFrame):
                vals = values[comp].to_numpy()
            elif len(cols) == 1:
                flat = np.asarray(
                    values.to_numpy() if isinstance(values, pd.Series) else values
                )
                vals = flat.ravel() if flat.ndim > 1 else flat
            else:
                flat = np.asarray(
                    values.to_numpy() if isinstance(values, pd.Series) else values
                )
                vals = flat[i] if pd.api.types.is_scalar(row_key) else flat[:, i]
            _write_to_chunks(self._view._slices[comp], pos_in_view, vals)
