import pandas as pd


class Component:

    def __init__(self, *fields):
        self.fields = fields
        for k in fields:
            # TODO: this is lame
            setattr(self, k,  k)
        return

    def init_dataframe(self):
        return pd.DataFrame(columns=self.fields)

    def __invert__(self):
        return ('not', self)
