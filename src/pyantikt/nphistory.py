import numpy as np

from pyantikt.pseudojet import PseudoJet

'''Jet merging history as numpy arrays'''
class NPHistory:
    def __init__(self, size:int):
        '''Initialise a history struture of arrays'''
        self.size = size

        # Counter for the next slot to fill, which is equivalent to the active 'size'
        # of the history
        self.next = 0

        # Index in history where first parent of this jet was created (-1 if this jet is an
        # original particle)
        self.parent1 = np.empty(size, dtype=int)
        self.parent1.fill(-1)

        # Index in history where second parent of this jet was created (-1 if this jet is an
        # original particle); BeamJet if this history entry just labels the fact that the jet has recombined
        # with the beam)
        self.parent2 = np.empty(size, dtype=int)
        self.parent2.fill(-1)

        # Index in history where the current jet is recombined with another jet to form its child. It
        # is -1 if this jet does not further recombine
        self.child = np.empty(size, dtype=int)
        self.child.fill(-1)

        # Index in the _jets vector where we will find the Jet object corresponding to this jet
        # (i.e. the jet created at this entry of the history). NB: if this element of the history
        # corresponds to a beam recombination, then jetp_index=Invalid
        self.jetp_index = np.empty(size, dtype=int)
        self.jetp_index.fill(-1)

        # The distance corresponding to the recombination at this stage of the clustering.
        self.dij = np.zeros(size, dtype=float)

        # The largest recombination distance seen so far in the clustering history.
        self.max_dij_so_far = np.zeros(size, dtype=float)

    def append(self, parent1: int, parent2:int, jetp_index:int, dij:float, max_dij_so_far:float):
        '''Append a new item to the history'''
        if self.next == self.size:
            raise RuntimeError("History structure is now full, cannot append")
        
        self.parent1[self.next] = parent1
        self.parent2[self.next] = parent2
        self.jetp_index[self.next] = jetp_index
        self.dij[self.next] = dij
        self.max_dij_so_far[self.next] = max_dij_so_far

        self.next += 1

    def fill_initial_history(self, jets:list[PseudoJet]) -> float:
        '''Fill the initial history with source jets'''
        Qtot = 0.0
        for ijet, jet in enumerate(jets):
            self.jetp_index[ijet] = ijet
            jet.cluster_hist_index = ijet
            Qtot = jet.E
        
        self.next = len(jets)

        return Qtot
