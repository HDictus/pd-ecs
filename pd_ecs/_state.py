import pandas as pd


class WorldState(dict):

    def give(self, transitions, components):
        """Triggered on give operation.
        Arguments:
            transitions: iterator of (new, old), eids
              giving entity ids for previous and new archetypes
            components: dataframe of new component values
        """
        # TODO: to implement observer pattern, could consider
        # making change-archetype and give different events
        # where the archetype manager will emit change-archetype
        # before give is resolved
        for (oldat, newat), eids in transitions:
            if oldat == newat:
                self[newat].loc[eids, components.columns] = components.loc[eids]
                continue
            new_state = pd.concat([
                self[oldat].loc[eids],
                components.loc[eids]
            ], axis=1)
            if newat not in self:
                self[newat] = new_state
            else:
                self[newat] = pd.concat([self[newat], new_state], axis=0)
            self[oldat] = self[oldat].drop(eids)

    def take(self, transitions):
        for (oldat, newat), eids in transitions:
            oldstate = self[oldat].loc[eids]
            if newat in self:
                self[newat] = pd.concat([self[newat], oldstate], axis=0)
            else:
                self[newat] = oldstate
            self[oldat] = self[oldat].drop(eids)
