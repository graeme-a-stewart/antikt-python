'''Structure of arrays container for holding numpy arrays that correspond to pseudojets'''

import numpy as np
import numpy.ma as ma

from numba import njit

from pyantikt.pseudojet import PseudoJet

class NPPseudoJets:
    def __init__(self, size:int=0):
        '''Setup blank arrays that will be filled later'''
        self.size = size
        self.next_slot = 0
        self.phi = ma.masked_array(np.zeros(size, dtype=float), np.zeros(size))
        self.rap = ma.masked_array(np.zeros(size, dtype=float), np.zeros(size))
        self.inv_pt2 = ma.masked_array(np.zeros(size, dtype=float), np.zeros(size))
        self.dist = ma.masked_array(np.zeros(size, dtype=float), np.zeros(size))
        self.nn = ma.masked_array(np.zeros(size, dtype=int), np.zeros(size)) # Index of my nearest neighbour
    
    def set_jets(self, jets:list[PseudoJet]):
        if len(jets) > self.phi.size:
            raise RuntimeError(f"Attempted to fill NP PseudoJets, but containers are too small ({self.size})")
        for ijet, jet in enumerate(jets):
            self.phi[ijet] = jet.phi
            self.rap[ijet] = jet.rap
            self.inv_pt2[ijet] = jet.inv_pt2
            self.nn[ijet] = -1
            self.dist[ijet] = 1e9
        for ijet in range(len(jets), self.phi.size):
            self.phi[ijet] = self.rap[ijet] = self.inv_pt2[ijet] = self.dist[ijet] = self.nn[ijet] = ma.masked
        self.next_slot = len(jets)

    def __str__(self):
        _string = ""
        for ijet in range(self.phi.size):
            _string += f"{ijet} - {self.phi[ijet]} {self.rap[ijet]} {self.inv_pt2[ijet]} {self.dist[ijet]} {self.nn[ijet]}\n"
        return _string
    
    def mask_slot(self, ijet: int):
        self.phi[ijet] = self.rap[ijet] = self.inv_pt2[ijet] = self.dist[ijet] = self.nn[ijet] = ma.masked

    def insert_jet(self, jet: PseudoJet, slot:int = -1) -> int:
        '''Add a new pseudojet into the numpy structures'''
        if (slot == -1):
            if self.next_slot == self.size:
                raise RuntimeError(f"Numpy jet containers are full (size {self.size})")
            slot = self.next_slot
            self.next_slot += 1
        # Setting values in these arrays also makes them valid (i.e., not masked)
        self.phi[slot] = jet.phi
        self.rap[slot] = jet.rap
        self.inv_pt2[slot] = jet.inv_pt2
        self.nn[slot] = -1
        self.dist[slot] = 1e9 # Necessary?

        return slot
    