import pandas as pd
import numpy as np


class EntityCollection(pd.DataFrame):
    """Represents multiple components of entities as a single dataframe.

    Mutating this dataframe propagates the changes to the parent dataframes.
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
