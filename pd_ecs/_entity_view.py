import numpy as np
import pandas as pd


class EntityView:
    """
    Lazy view over entities matching a multi-component filter.

    .index is computed eagerly; component columns are materialized on demand
    by slicing the backing numpy arrays of each component's Series.
    """

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

    def _backing_positions(self, comp):
        """Array mapping each view position -> position in comp's backing array."""
        result = np.empty(len(self.index), dtype=np.intp)
        offset = 0
        for _, start, stop in self._slices[comp]:
            n = stop - start
            result[offset:offset + n] = np.arange(start, stop)
            offset += n
        return result

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
        return pd.DataFrame(
            {comp: self._view[comp].loc[key] for comp in self._view._slices},
            index=pd.Index(key),
        )

    def __setitem__(self, key, values):
        if not isinstance(key, tuple):
            raise ValueError("loc assignment requires (row_key, col_key)")
        row_key, col_key = key
        cols = col_key if isinstance(col_key, list) else [col_key]

        if pd.api.types.is_scalar(row_key):
            pos_in_view = np.array([self._view.index.get_loc(row_key)])
        else:
            pos_in_view = self._view.index.get_indexer(row_key)

        for i, comp in enumerate(cols):
            target = self._view._backing_positions(comp)[pos_in_view]
            backing = self._view._slices[comp][0][0]
            if isinstance(values, pd.DataFrame):
                backing[target] = values[comp].to_numpy()
            elif len(cols) == 1:
                flat = np.asarray(
                    values.to_numpy() if isinstance(values, pd.Series) else values
                )
                backing[target] = flat.ravel() if flat.ndim > 1 else flat
            else:
                flat = np.asarray(
                    values.to_numpy() if isinstance(values, pd.Series) else values
                )
                backing[target] = flat[i] if pd.api.types.is_scalar(row_key) else flat[:, i]
