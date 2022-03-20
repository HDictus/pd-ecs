import pandas as pd


class Component:

    def __init__(self, *fields):
        self.fields = fields
        return

    def init_dataframe(self):
        return pd.DataFrame(columns=self.fields, dtype=int)
