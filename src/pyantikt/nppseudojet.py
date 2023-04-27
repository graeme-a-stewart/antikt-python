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
        self.phi = np.zeros(size, dtype=float)
        self.rap = np.zeros(size, dtype=float)
        self.inv_pt2 = np.zeros(size, dtype=float)
        self.dist = np.zeros(size, dtype=float)
        self.akt_dist = np.zeros(size, dtype=float)
        self.nn = np.zeros(size, dtype=int) # Index of my nearest neighbour
        self.mask = np.ones(size, dtype=bool)
    
    def set_jets(self, jets:list[PseudoJet]):
        if len(jets) > self.phi.size:
            raise RuntimeError(f"Attempted to fill NP PseudoJets, but containers are too small ({self.size})")
        for ijet, jet in enumerate(jets):
            self.phi[ijet] = jet.phi
            self.rap[ijet] = jet.rap
            self.inv_pt2[ijet] = jet.inv_pt2
            self.nn[ijet] = -1
            self.dist[ijet] = self.akt_dist[ijet] = 1e20
            self.mask[ijet] = False
        self.next_slot = len(jets)
        self.dist[len(jets):] = self.akt_dist[len(jets):] = 1e20

    def __str__(self):
        _string = ""
        for ijet in range(self.phi.size):
            _string += (f"{ijet} - {self.phi[ijet]} {self.rap[ijet]} {self.inv_pt2[ijet]} {self.dist[ijet]} "
            f"{self.akt_dist[ijet]} {self.nn[ijet]} "
            f"(mask: {self.mask[ijet]})\n")
        return _string
    
    def mask_slot(self, ijet: int):
        self.mask[ijet] = True
        self.dist[ijet] = self.akt_dist[ijet] = 1e20

    def insert_jet(self, jet: PseudoJet, slot:int = -1) -> int:
        '''Add a new pseudojet into the numpy structures'''
        if (slot == -1):
            if self.next_slot == self.size:
                raise RuntimeError(f"Numpy jet containers are full (size {self.size})")
            slot = self.next_slot
            self.next_slot += 1
        self.phi[slot] = jet.phi
        self.rap[slot] = jet.rap
        self.inv_pt2[slot] = jet.inv_pt2
        self.nn[slot] = -1
        self.dist[slot] = self.akt_dist[slot] = 1e20 # Necessary?
        self.mask[slot] = False

        return slot
    
    def print_jet(self, ijet:int):
        return (f"{ijet} - {self.phi[ijet]} {self.rap[ijet]} {self.inv_pt2[ijet]} "
        f"{self.dist[ijet]} {self.akt_dist[ijet]} {self.nn[ijet]} "
        f"(mask: {self.mask[ijet]} -> {self.mask[self.nn[ijet]] if self.nn[ijet] >= 0 else None})")
    