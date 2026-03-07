import pandas as pd
import numpy as np


class EntityCollectionLocAccessor:
    """Custom .loc accessor that propagates writes to underlying DataFrames."""

    def __init__(self, obj):
        self._obj = obj

    def __getitem__(self, key):
        return self._obj._loc_getitem(key)

    def __setitem__(self, key, value):
        # First, set in the combined dataframe using pandas' loc
        pd.DataFrame.loc.fget(self._obj)[key] = value

        # Parse the key to extract row and column selectors
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key = key
            col_key = slice(None)

        # Propagate to underlying dataframes
        obj = self._obj
        if hasattr(obj, '_dataframes') and obj._dataframes:
            for df in obj._dataframes:
                # Determine which columns to update
                if isinstance(col_key, str):
                    cols = [col_key] if col_key in df.columns else []
                elif isinstance(col_key, list):
                    cols = [c for c in col_key if c in df.columns]
                elif isinstance(col_key, slice) and col_key == slice(None):
                    cols = [c for c in obj.columns if c in df.columns]
                else:
                    cols = [c for c in df.columns if c in obj.columns]

                if not cols:
                    continue

                # Get the intersection of row indices
                if isinstance(row_key, slice) and row_key == slice(None):
                    row_idx = obj.index.intersection(df.index)
                elif isinstance(row_key, (list, pd.Index, np.ndarray)):
                    row_idx = pd.Index(row_key).intersection(df.index)
                else:
                    row_idx = pd.Index([row_key]).intersection(df.index)

                if len(row_idx) == 0:
                    continue

                # Propagate the value
                for col in cols:
                    df.loc[row_idx, col] = pd.DataFrame.loc.fget(obj)[row_idx, col]


class EntityCollection(pd.DataFrame):
    """Represents multiple components of entities as a single dataframe.

    Mutating this dataframe propagates the changes to the parent datafframes
    """

    _metadata = ['_dataframes']

    def __init__(self, *dataframes, **kwargs):
        if dataframes:
            # Store references to the original dataframes
            dfs = list(dataframes)
            # Create the concatenated view using inner join
            combined = pd.concat(dfs, axis=1, join='inner')
            super().__init__(combined, **kwargs)
            self._dataframes = dfs
        else:
            super().__init__(**kwargs)
            self._dataframes = []

    @property
    def _constructor(self):
        """Return the constructor for creating new instances."""
        def _c(*args, **kwargs):
            result = EntityCollection(*args, **kwargs)
            # Preserve dataframes reference if not explicitly provided
            if not result._dataframes and hasattr(self, '_dataframes'):
                result._dataframes = self._dataframes
            return result
        return _c

    @property
    def loc(self):
        """Return custom .loc accessor that propagates writes."""
        return EntityCollectionLocAccessor(self)

    def _loc_getitem(self, key):
        """Delegate read access to pandas' loc."""
        return pd.DataFrame.loc.fget(self)[key]

    def __setitem__(self, key, value):
        """Set column value and propagate to underlying DataFrames."""
        # Set in the combined dataframe first
        super().__setitem__(key, value)

        # Propagate to underlying dataframes that already have this column
        if hasattr(self, '_dataframes') and self._dataframes:
            for df in self._dataframes:
                if key not in df.columns:
                    continue
                # Get the intersection of indices
                common_idx = self.index.intersection(df.index)
                if len(common_idx) > 0:
                    # Get the value for the common indices
                    if isinstance(value, (pd.Series, np.ndarray)):
                        # Handle array-like values - align by index
                        if isinstance(value, pd.Series):
                            df.loc[common_idx, key] = value.loc[common_idx]
                        else:
                            df.loc[common_idx, key] = self.loc[common_idx, key]
                    else:
                        # Scalar value
                        df.loc[common_idx, key] = value
